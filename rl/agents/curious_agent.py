from __future__ import division
import warnings
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense
from keras.utils.generic_utils import Progbar
from keras.losses import mean_squared_error, categorical_crossentropy

import sys
sys.path.append('../../')

from rl.core import Agent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.util import *
from rl.memory import PrioritizedMemory, PartitionedMemory
from rl.layers import *
from rl.agents.dqn import AbstractDQNAgent, mean_q

from keras.utils import to_categorical

import pdb


# An modification (by Badlani, Villa) of the DQN implementation as described in Mnih (2013) and Mnih (2015) to incorporate Curiosity (Pathak et al).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class CuriousDQNAgent(AbstractDQNAgent):
    """
    # Arguments
        model__: A Keras model.
        curiosity_forward_model: Curiosity Fwd Keras model.
        curiosity_inverse_model: Curiosity Inverse Keras model.
        policy__: A Keras-rl policy that are defined in [policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py).
        test_policy__: A Keras-rl policy.
        enable_double_dqn__: A boolean which enables target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        enable_dueling_dqn__: A boolean which enables dueling architecture proposed by Wang et al.
        dueling_type__: If `enable_dueling_dqn` is set to `True`, a type of dueling architecture must be chosen which calculate Q(s,a) from V(s) and A(s,a) differently. Note that `avg` is recommanded in the [paper](https://arxiv.org/abs/1511.06581).
            `avg`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            `max`: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            `naive`: Q(s,a;theta) = V(s;theta) + A(s,a;theta)
        nb_actions__: The total number of actions the agent can take. Dependent on the environment.
        processor__: A Keras-rl processor. An intermediary between the environment and the agent. Resizes the input, clips rewards etc. Similar to gym env wrappers.
        nb_steps_warmup__: An integer number of random steps to take before learning begins. This puts experience into the memory.
        gamma__: The discount factor of future rewards in the Q function.
        target_model_update__: How often to update the target model. Longer intervals stabilize training.
        train_interval__: The integer number of steps between each learning process.
        delta_clip__: A component of the huber loss.
    """
    def __init__(self, model, curiosity_forward_model=None, curiosity_inverse_model=None, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(CuriousDQNAgent, self).__init__(*args, **kwargs)


        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
        if model.output._keras_shape != (None, self.nb_actions):
            raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            model = self._build_dueling_arch(model, dueling_type)
        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyQPolicy()
        if test_policy is None:
            test_policy = GreedyQPolicy()
        self.policy = policy
        self.test_policy = test_policy

        self.reset_states()

        #flag for changes to algorithm that come from dealing with importance sampling weights and priorities
        self.prioritized = True if isinstance(self.memory, PrioritizedMemory) else False

        self.curiosity_forward_model = curiosity_forward_model
        self.curiosity_inverse_model = curiosity_inverse_model
        
    def get_config(self):
        config = super(DQNAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        config['curiosity_forward_model'] = get_object_config(self.curiosity_forward_model)
        config['curiosity_inverse_model'] = get_object_config(self.curiosity_inverse_model)

        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        ## We never train these models, hence we can set the optimizer and loss arbitrarily
        if self.curiosity_forward_model != None:
            self.curiosity_forward_model.compile(optimizer='sgd', loss='mse')

        if self.curiosity_inverse_model != None:
            self.curiosity_inverse_model.compile(optimizer='sgd', loss='categorical_crossentropy')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, importance_weights, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            # adjust updates by importance weights. Note that importance weights are just 1.0
            # (and have no effect) if not using a prioritized memory
            return K.sum(loss * importance_weights, axis=-1)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        importance_weights = Input(name='importance_weights',shape=(self.nb_actions,))
        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, importance_weights, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        
        #inputs to curiosity models
        ins_curiosity_forward = [self.curiosity_forward_model.input] if type(self.curiosity_forward_model.input) is not list else self.curiosity_forward_model.input
        ins_curiosity_inverse = [self.curiosity_inverse_model.input] if type(self.curiosity_inverse_model.input) is not list else self.curiosity_inverse_model.input
        #end inputs

        #outputs from curiosity models
        curiousity_forward_out = self.curiosity_forward_model.output
        curiosity_inverse_out = self.curiosity_inverse_model.output
        #end outputs

        trainable_model = Model(inputs=ins + [y_true, importance_weights, mask] + ins_curiosity_forward + ins_curiosity_inverse, outputs=[loss_out, y_pred, curiosity_forward_out, curiosity_inverse_out])
        
        #Commenting out since now we have 2 additional outputs
        #assert len(trainable_model.output_names) == 2

        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
            mean_squared_error,
            categorical_crossentropy
        ]
        #for now, loss is a straight sum of all losses; note that the second loss is always 0; we don't use it
        trainable_model.compile(optimizer=optimizer, loss=losses, loss_weights=[1.0,1.0,1.0,1.0], metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

        if self.curiosity_forward_model != None:
            self.curiosity_forward_model.load_weights(filepath + "_curiosity_forward")
        if self.curiosity_inverse_model != None:
            self.curiosity_inverse_model.load_weights(filepath + "_curiosity_inverse")

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)
        if self.curiosity_forward_model != None:
            self.curiosity_forward_model.save_weights(filepath + "_curiosity_forward", overwrite=overwrite)
        if self.curiosity_inverse_model != None:
            self.curiosity_inverse_model.save_weights(filepath + "_curiosity_inverse", overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

            if self.curiosity_forward_model != None:
                self.curiosity_forward_model.reset_states()
            if self.curiosity_inverse_model != None:
                self.curiosity_inverse_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:

            if self.prioritized:
                # Calculations for current beta value based on a linear schedule.
                current_beta = self.memory.calculate_beta(self.step)
                # Sample from the memory.
                experiences = self.memory.sample(self.batch_size, current_beta)
            else:
                #SequentialMemory
                experiences = self.memory.sample(self.batch_size)

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            importance_weights = []
            # We will be updating the idxs of the priority trees with new priorities
            pr_idxs = []

            if self.prioritized:
                for e in experiences[:-2]: # Prioritized Replay returns Experience tuple + weights and idxs.
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)
                importance_weights = experiences[-2]
                pr_idxs = experiences[-1]
            else: #SequentialMemory
                for e in experiences:
                    state0_batch.append(e.state0)
                    state1_batch.append(e.state1)
                    reward_batch.append(e.reward)
                    action_batch.append(e.action)
                    terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                q_batch = self._calc_double_q_values(state1_batch)
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                assert target_q_values.shape == (self.batch_size, self.nb_actions)
                q_batch = np.max(target_q_values, axis=1).flatten()
            assert q_batch.shape == (self.batch_size,)

            targets = np.zeros((self.batch_size, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * q_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            #Putting together the multi-step target
            Rs = reward_batch + discounted_reward_batch

            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[action] = R  # update action with estimated accumulated reward
                dummy_targets[idx] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            if not self.prioritized:
                importance_weights = [1. for _ in range(self.batch_size)]
            #Make importance_weights the same shape as the other tensors that are passed into the trainable model
            assert len(importance_weights) == self.batch_size
            importance_weights = np.array(importance_weights)
            importance_weights = np.vstack([importance_weights]*self.nb_actions)
            importance_weights = np.reshape(importance_weights, (self.batch_size, self.nb_actions))
            # Perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            ins = [state0_batch] if type(self.model.input) is not list else state0_batch
            
            encoded_actions = to_categorical(action_batch, num_classes=self.nb_actions)
            metrics = self.trainable_model.train_on_batch(ins + [targets, importance_weights, masks] + [state0_batch,encoded_actions] + [state0_batch, state1_batch], [dummy_targets, targets, state1_batch, encoded_actions])


            ########### TRAIN INVERSE AND FORWARD MODELS ##############
            ## inverse model ##
            """
            ##need to figure out if action batch is 1 hot or simply action number
            if self.curiosity_forward_model is not None or self.curiosity_inverse_model is not None:

                if self.curiosity_inverse_model is not None:
                    self.curiosity_inverse_model.train_on_batch(x=[state0_batch, state1_batch], y=encoded_actions)

                ##forward model ##
                if self.curiosity_forward_model is not None:
                    self.curiosity_forward_model.train_on_batch(x=[state0_batch, encoded_actions],y=state1_batch)
            """
            ########### END TRAIN INVERSE AND FORWARD MODELS ###########

            if self.prioritized:
                assert len(pr_idxs) == self.batch_size
                #Calculate new priorities.
                y_true = targets
                y_pred = self.model.predict_on_batch(ins)
                #Proportional method. Priorities are the abs TD error with a small positive constant to keep them from being 0.
                new_priorities = (abs(np.sum(y_true - y_pred, axis=-1))) + 1e-5
                assert len(new_priorities) == self.batch_size
                #update priorities
                self.memory.update_priorities(pr_idxs, new_priorities)

            metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)

