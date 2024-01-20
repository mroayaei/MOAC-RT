import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import regularizers
# Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
# TensorFlow to build a static graph out of the logic and computations in our function.
# This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.


@tf.function
def update(
    state_batch, action_batch, reward_batch, next_state_batch,
    target_actor, target_critic, critic_model, actor_model,
    critic_optimizer, actor_optimizer,
    gamma
):
    # Training and updating Actor & Critic networks.
    # See Pseudo Code.
    with tf.GradientTape() as tape:
        target_actions = target_actor(next_state_batch, training=True)
        y = reward_batch + gamma * target_critic(
            [next_state_batch, target_actions], training=True
        )
        critic_value = critic_model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

    critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, critic_model.trainable_variables)
    )

    with tf.GradientTape() as tape:
        actions = actor_model(state_batch, training=True)
        critic_value = critic_model([state_batch, actions], training=True)
        # Used `-value` as we want to maximize the value given
        # by the critic for our actions
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, actor_model.trainable_variables)
    )

# We compute the loss and update parameters
def learn(buffer,
            target_actor, target_critic, critic_model, actor_model,
            critic_optimizer, actor_optimizer,
            gamma):
    # Get sampling range
    record_range = min(buffer.buffer_counter, buffer.buffer_capacity)
    # Randomly sample indices
    batch_indices = np.random.choice(record_range, buffer.batch_size)

    # Convert to tensors
    state_batch = tf.convert_to_tensor(buffer.state_buffer[batch_indices])
    action_batch = tf.convert_to_tensor(buffer.action_buffer[batch_indices])
    reward_batch = tf.convert_to_tensor(buffer.reward_buffer[batch_indices])
    reward_batch = tf.cast(reward_batch, dtype=tf.float32)
    next_state_batch = tf.convert_to_tensor(buffer.next_state_buffer[batch_indices])

    update(state_batch, action_batch, reward_batch, next_state_batch,
            target_actor, target_critic, critic_model, actor_model,
            critic_optimizer, actor_optimizer,
            gamma)

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))



wdir = "../DDPG_DENSNET/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

def get_actor():

    acti='linear'
    inputs = layers.Input(shape=(50,50,3))
    out = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-1,1))(inputs)
    out = tf.keras.layers.experimental.preprocessing.RandomFlip()(out)
    out = tf.keras.applications.mobilenet_v2.preprocess_input(out)
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=wdir,include_top=False)
    model.trainable=True
    out = model(out)

    out = layers.Flatten()(out)

    out = layers.Dense(256, activation='linear')(out)
    out = layers.Dense(256, activation='linear')(out)
    
    output_1 = layers.Dense(1, activation="sigmoid", activity_regularizer=tf.keras.regularizers.L2(0.001),
                            kernel_constraint=tf.keras.constraints.max_norm(0.1))(out)
    output_2 = layers.Dense(1, activation="sigmoid",
                            kernel_constraint=tf.keras.constraints.max_norm(0.1),
                            activity_regularizer=tf.keras.regularizers.L2(0.001))(out)

    outputs = layers.Concatenate()([output_1, output_2])
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic():

    acti='linear'
    state_input = layers.Input(shape=(50,50,3))
    state_out = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-1,1))(state_input)
    state_out = tf.keras.layers.experimental.preprocessing.RandomFlip()(state_out)
    state_out = tf.keras.applications.mobilenet_v2.preprocess_input(state_out)
    model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=wdir,include_top=False)
    model.trainable=True
    state_out = model(state_out)

    state_out = layers.Flatten()(state_out)

    state_out = layers.Dense(256, activation='linear')(state_out)
    state_out = layers.Dense(256, activation='linear')(state_out)
    
    state_out = layers.Dense(16, activation=acti)(state_out)
    state_out = layers.Dense(32, activation=acti)(state_out)


    # Action as input
    action_input = layers.Input(shape=(2,))
    action_out = layers.Dense(32, activation=acti)(action_input)
    

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])
    

    out = layers.Dense(256, activation='linear')(concat)
    out = layers.Dense(256, activation='linear')(out)
    

    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def policy(actor_model, state, noise_object, cond=True):
    
    sampled_actions = tf.squeeze(actor_model(state)).numpy()
    if cond:
        noise1 = noise_object[0]()
        noise2 = noise_object[1]()
        
        # Adding noise to action
        sampled_actions[0] = sampled_actions[0] + noise1
        sampled_actions[1] = sampled_actions[1] + noise2

    # We make sure action is within bounds
    legal_action = []
    legal_action.append(np.clip(sampled_actions[0], 0., 1.))
    legal_action.append(np.clip(sampled_actions[1], 0., 1.))
    return legal_action, sampled_actions