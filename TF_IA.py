import tensorflow as tf
import numpy as np
import datetime
import os

############################                ######################################################
############################DISABLE DEBUG TF######################################################
import logging

tf.get_logger().setLevel(logging.ERROR)

#################################################################################################

####################################                             ################################
####################################SIMULATE METASPLOIT ENV STATE################################

##########################################     ##################################################
##########################################INDEX##################################################
OS_INDEX = 0
SERVICE_INDEX = 1
VERSION_INDEX = 2
EXPLOIT_INDEX = 3
TARGET_INDEX = 4

MY_NUMBER_OF_TEST = 100
##########################################         ##############################################
##########################################HASHTABLE##############################################
success_table = {
    '1': {
        'os': 'windows',
        'service': 'microsoft-ds',
        'version': 'any',
        'exploit': 'windows/smb/ms17_010_eternalblue',
        'target': 'win64',
        'payload': 'meterpreter_reverse_tcp'
    },
    '2': {
        'os': 'linux',
        'service': 'http',
        'version': 'any',
        'exploit': 'linux/http/pulse_secure_cmd_exec',
        'target': 'linux64',
        'payload': 'shell_java'
    }
}


#########################################           ########################################################
#########################################ENVIRONMENT########################################################

class Fake_environnement:
    def __init__(self):
        self.list_of_os = ['windows', 'linux']  # short_list
        self.list_of_service = ['microsoft-ds', 'http']
        self.list_of_version = ['any']  # versions fake
        self.list_of_exploit = ['windows/smb/ms17_010_eternalblue',
                                'linux/http/pulse_secure_cmd_exec', 'fuck']  # exploit short list
        self.list_of_target = ['win64', 'linux64']  # targets fake
        self.list_of_payload = ['meterpreter_reverse_tcp', 'shell_java']  # payload fake
        self.current_state = []
        self.number_of_sample = 0
        self.background_color_monitor = Background_printer()

    def get_state(self):  #####simulate the get of a state
        self.reset_state()
        cursor_os = self.list_of_os.index(np.random.choice(self.list_of_os))
        cursor_service = cursor_os
        cursor_version = 0  # any anyway
        cursor_exploit = cursor_os
        cursor_target = cursor_os
        ######################################################implement those famous state
        self.current_state.insert(OS_INDEX, cursor_os)
        self.current_state.insert(SERVICE_INDEX, cursor_service)
        self.current_state.insert(VERSION_INDEX, cursor_version)
        self.current_state.insert(EXPLOIT_INDEX, cursor_exploit)
        self.current_state.insert(TARGET_INDEX, cursor_target)
        return self.current_state

    def reset_state(self):  ######reset the state
        self.current_state = []

    def simulate_a_try(self, payload):  ######simulate the success (according to the hash table)
        for key, dict in success_table.items():
            if dict.get('os') == self.list_of_os[self.current_state[OS_INDEX]] and dict.get('service') == \
                    self.list_of_service[self.current_state[
                        SERVICE_INDEX]] and (
                    dict.get('version') == self.list_of_version[self.current_state[VERSION_INDEX]] or
                    self.list_of_version[self.current_state[VERSION_INDEX]] == 'any') \
                    and dict.get('exploit') == self.list_of_exploit[self.current_state[EXPLOIT_INDEX]] and \
                    dict.get('target') == self.list_of_target[self.current_state[TARGET_INDEX]] and \
                    dict.get('payload') == payload:
                print(env.background_color_monitor.background_OKGREEN + "[*] current state : ", self.current_state,
                      env.background_color_monitor.background_ENDC)
                print(env.background_color_monitor.background_OKGREEN + "[*] success with : ",
                      self.list_of_os[self.current_state[OS_INDEX]],
                      self.list_of_service[self.current_state[SERVICE_INDEX]],
                      self.list_of_version[self.current_state[VERSION_INDEX]],
                      self.list_of_exploit[self.current_state[EXPLOIT_INDEX]],
                      self.list_of_target[self.current_state[TARGET_INDEX]],
                      "||| payload chosen",
                      payload,
                      env.background_color_monitor.background_ENDC)
                return True
        print(env.background_color_monitor.background_FAIL + "[x] failure with : ",
              self.list_of_os[self.current_state[OS_INDEX]],
              self.list_of_service[self.current_state[SERVICE_INDEX]],
              self.list_of_version[self.current_state[VERSION_INDEX]],
              self.list_of_exploit[self.current_state[EXPLOIT_INDEX]],
              self.list_of_target[self.current_state[TARGET_INDEX]],
              "||| payload chosen",
              payload,
              env.background_color_monitor.background_ENDC)
        return False

    def make_a_move(self, payload,
                    number_of_test=MY_NUMBER_OF_TEST):  ######################simulate the try of launching the payload
        self.number_of_sample += 1
        flag_success = self.simulate_a_try(payload)
        if flag_success:
            reward = 200
        else:
            reward = -200
        # if self.number_of_sample > number_of_test:
        done = True
        # else:
        #     done = False
        # create next state
        return self.get_state(), reward, done


class Background_printer:

    def __init__(self):
        self.background_HEADER = '\033[95m'
        self.backgrounf_OKBLUE = '\033[94m'
        self.background_OKCYAN = '\033[96m'
        self.background_OKGREEN = '\033[92m'
        self.background_WARNING = '\033[93m'
        self.background_FAIL = '\033[91m'
        self.background_ENDC = '\033[0m'
        self.background_BOLD = '\033[1m'
        self.background_UNDERLINE = '\033[4m'


#######################################      #######################################################
#######################################IA LAB#######################################################
class Local_brain:

    def __init__(self, num_of_states, num_of_actions):
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.model = self._build_model()
        self.color_monitor = Background_printer()

    def __call__(self, inputs):
        out_actions, out_value = self.model(inputs)
        return out_actions, out_value

    def _build_model(self):
        ##Model of the IA brain
        inputs = tf.keras.layers.Input(batch_shape=(None, 5))
        l_dense1 = tf.keras.layers.Dense(50, activation='relu')(inputs)
        l_dense2 = tf.keras.layers.Dense(100, activation='relu')(l_dense1)
        l_dense3 = tf.keras.layers.Dense(200, activation='relu')(l_dense2)
        l_dense4 = tf.keras.layers.Dense(400, activation='relu')(l_dense3)
        out_actions = tf.keras.layers.Dense(self.num_of_actions, activation='softmax')(l_dense4)
        out_value = tf.keras.layers.Dense(1, activation='linear')(l_dense4)
        model = tf.keras.models.Model(inputs=[inputs], outputs=[out_actions, out_value])

        # model.make_predict_function()
        return model

    def prediction(self, valuevector):
        try:
            # print(self.color_monitor.background_OKGREEN + "[*] valuevector is : ", valuevector,
            #       self.color_monitor.background_ENDC)
            return self.model.predict(np.atleast_2d(valuevector))  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        except:
            print(
                self.color_monitor.background_WARNING + "[!] Maybe bad shape of the value vector" + self.color_monitor.background_ENDC)
            # print(self.color_monitor.background_OKGREEN + "[*] valuevector is : ", valuevector,
            #       self.color_monitor.background_ENDC)
            return self.model.predict(valuevector)

    def get_trainable_variables(self):
        return self.model.trainable_variables

    def saver_weights(self, filepath):
        return self.model.save_weights(filepath)

    def loader_weights(self, filepath):
        return self.model.load_weights(filepath)


class DQN:

    def __init__(self, num_of_states, num_of_actions, gamma, max_experiences, min_experiences, batch_size,
                 learning_rate):
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = tf.optimizers.RMSprop(learning_rate)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [],
                           'done': []}  # Q(s, a) = max(r + Q(s2, a)) Bellman equation's buffer
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.model = Local_brain(num_of_states, num_of_actions)
        self.color_monitor = Background_printer()

    def train(self, TargetNet):  # train the TrainNet DQN
        if len(self.experience['s']) < self.min_experiences:
            return 0

        states = np.asarray(self.experience['s'])
        actions = np.asarray(self.experience['a'])
        rewards = np.asarray(self.experience['r'])
        states_next = np.asarray(self.experience['s2'])
        dones = np.asarray(self.experience['done'])  # indicate if s is the last state or not

        value_next = np.max(TargetNet.model.prediction(np.atleast_2d(states_next))[0], axis=1)  # predict the next value
        actual_values = np.where(dones, rewards,
                                 rewards + self.gamma * value_next)  # grounded values, opposed to predict value

        with tf.GradientTape(watch_accessed_variables=True) as tape:
            selected_action_values = tf.math.reduce_sum(
                self.model(np.atleast_2d(states))[0] * tf.one_hot(actions, self.num_of_actions), axis=1)
            # manually mask the logits #####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # self.model(entry) here cause we want a tensor not a numpy, or gradient won't work

            loss = tf.math.reduce_mean(tf.math.reduce_mean(tf.square(
                actual_values - selected_action_values)))  # mean of mean of square loss of the real target and prediction (double mean to improve gap between  value)
            tape.watch(loss)

            variables = self.model.get_trainable_variables()
            tape.watch(variables)

            gradients = tape.gradient(loss, variables)

        self.optimizer.apply_gradients(zip(gradients,
                                           variables))  # backpropagation to other layer (zip takes two iterable and return a list of tuple)
        return loss

    def get_action(self, states,
                   epsilon):  # use of epsilon greedy => at the beginning we explore random actions and the more we get action, the more we get focus on the model
        print(self.color_monitor.background_OKGREEN + '[*] value of epsilon : ', epsilon,
              self.color_monitor.background_ENDC)
        if epsilon > 0.1:
            return np.random.choice(self.num_of_actions)
        else:
            return np.argmax(self.model.prediction(np.atleast_2d(states))[0])

        # get the prob of all action and get the most efficient ###################
        ###########################################################################

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables_1 = self.model.get_trainable_variables()
        variables_2 = TrainNet.model.get_trainable_variables()
        for v1, v2 in zip(variables_1, variables_2):
            v1.assign(v2.numpy())

    def save_weights(self, filepath):
        return self.model.saver_weights(filepath)


############################################                          #####################################################
############################################SIMULATION OF EXPLOITATION#####################################################

# TODO
# implement in the main_project and cable it

def exploit(env, TrainNet, TargetNet, epsilon, copy_step):
    # iter = 0
    done = False
    losses = list()
    states = env.get_state()
    while not done:
        action = env.list_of_payload[
            TrainNet.get_action(states, epsilon)]  # get the payload according to epsilon greedy
        previous_states = states  # save previous state
        states, reward, done = env.make_a_move(action)  # make a move

        ####convert action to index
        action = env.list_of_payload.index(action)

        if done:
            env.reset_state()

        exp = {'s': previous_states, 'a': action, 'r': reward, 's2': states, 'done': done}
        TrainNet.add_experience(exp)

        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        # iter += 1
        # if iter % copy_step == 0:
        TargetNet.copy_weights(TrainNet)
    return reward, np.mean(losses)


if __name__ == '__main__':

    env = Fake_environnement()
    gamma = 0.99
    copy_step = 25
    num_states = 5
    num_actions = len(env.list_of_payload)
    max_experiences = MY_NUMBER_OF_TEST
    min_experiences = 0
    batch_size = 32
    learning_rate = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'log_dir/log' + current_time
    data_saver_dir = 'trained_data/cp.cpkt'  # create the saver
    flag_epsilon = False

    # create the DQN Agents
    TrainNet = DQN(num_states, num_actions, gamma, max_experiences, min_experiences, batch_size, learning_rate)
    TargetNet = DQN(num_states, num_actions, gamma, max_experiences, min_experiences, batch_size, learning_rate)

    #########################PRINT BANNER
    print(env.background_color_monitor.background_OKGREEN + u"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
██╗  ██╗ █████╗  ██████╗██╗  ██╗    ████████╗ ██████╗  ██████╗ ██╗     ██████╗  ██████╗ ██╗  ██╗
██║  ██║██╔══██╗██╔════╝██║ ██╔╝    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     ██╔══██╗██╔═══██╗╚██╗██╔╝
███████║███████║██║     █████╔╝        ██║   ██║   ██║██║   ██║██║     ██████╔╝██║   ██║ ╚███╔╝ 
██╔══██║██╔══██║██║     ██╔═██╗        ██║   ██║   ██║██║   ██║██║     ██╔══██╗██║   ██║ ██╔██╗ 
██║  ██║██║  ██║╚██████╗██║  ██╗       ██║   ╚██████╔╝╚██████╔╝███████╗██████╔╝╚██████╔╝██╔╝ ██╗
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝   (Lotus version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    """ + env.background_color_monitor.background_ENDC)

    ##test or train variable
    train_or_test = input(
        env.background_color_monitor.background_HEADER + '[!!!] train or test : ' + env.background_color_monitor.background_ENDC)

    if train_or_test == 'train':
        print(
            env.background_color_monitor.background_OKCYAN + "[*] Training ..." + env.background_color_monitor.background_ENDC)

        ###############################                                                       #################################
        ###############################TRY FIRST TO FIND OUT IF THERE IS EXISTING TRANING DATA#################################

        if os.path.exists(data_saver_dir + '.index') is True:
            print(env.background_color_monitor.background_OKGREEN + u"""
      ______ ______
    _/      Y      \_
   // ~~ ~~ | ~~ ~  \\
  // ~ ~ ~~ | ~~~ ~~ \\
 //________.|.________\\ 
`----------`-'----------' [*] Restore learned data
            """ + env.background_color_monitor.background_ENDC)
            # load data
            TrainNet.model.loader_weights(data_saver_dir)
            # flag epsilon
            flag_epsilon = True

        # assign epsilon greedy
        if flag_epsilon:
            epsilon = 0.1
        else:
            epsilon = 0.99

        # proceed
        N = 5000
        total_rewards = []
        decay = 0.999
        min_epsilon = 0.1

        #create a summary for the log
        summary_writer = tf.summary.create_file_writer(log_dir)

        for n in range(N):
            epsilon = max(min_epsilon, epsilon * decay)
            total_reward, losses = exploit(env, TrainNet, TargetNet, epsilon, copy_step)
            total_rewards.append(total_reward)
            average_rewards = np.mean(total_rewards)

            with summary_writer.as_default():
                tf.summary.scalar('Episode reward', total_reward, step=n)
                tf.summary.scalar('Running avg reward(100)', average_rewards, step=n)
                tf.summary.scalar('Average loss', losses, step=n)
            if n % 100 == 0:
                print("\n\n")
                print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):",
                      average_rewards,
                      "episode loss: ", losses)
                print("\n\n")
        print(env.background_color_monitor.background_OKGREEN + "[*] Average reward for last 100 episodes:",
              average_rewards, env.background_color_monitor.background_ENDC)
        print(env.background_color_monitor.background_OKGREEN + "[*] Experience got : ", TrainNet.experience,
              env.background_color_monitor.background_ENDC)
        print(env.background_color_monitor.background_OKGREEN + "[*] Saving weight ...",
              env.background_color_monitor.background_ENDC)
        TrainNet.save_weights(data_saver_dir)
        
        ############################TEST THE IA#############################################################################
        ############################           #############################################################################
        
        print(env.background_color_monitor.background_OKCYAN + "[*] End" + env.background_color_monitor.background_ENDC)
        print(env.background_color_monitor.background_BOLD + """
ア マン カンノツ ウンデルスタンヅ ゼ アルツ ヘ イス スツヂイング イフ ヘ オンリ ロウク フオル ゼ エンヅ 
レズルツ ヰザオウツ タキング ゼ チメ ト =デレヴエ デウペリ イント ゼ レゾニング オフ ゼ スツヂ。

ミャモト ムサシ""" + env.background_color_monitor.background_ENDC)


    else:

        print(
            env.background_color_monitor.background_OKCYAN + "[*] Testing ..." + env.background_color_monitor.background_ENDC)
        print(
            env.background_color_monitor.background_OKGREEN + "[*] Loading value ..." + env.background_color_monitor.background_ENDC)

        # load data
        TrainNet.model.loader_weights(data_saver_dir)

        # see probability
        print(
            env.background_color_monitor.background_WARNING + "[!] Check the value by your self" + env.background_color_monitor.background_ENDC)
        val_1 = TrainNet.model.prediction([0, 0, 0, 0, 0])
        val_2 = TrainNet.model.prediction([1, 1, 0, 1, 1])
        print(
            env.background_color_monitor.background_OKGREEN + "\n------------------------------------------------------------------------------------------------------------------------------------" + env.background_color_monitor.background_ENDC)
        print(env.background_color_monitor.background_BOLD + "[*] For state [0 0 0 0 0] :", val_1,
              env.background_color_monitor.background_ENDC)
        print(env.background_color_monitor.background_BOLD + "[*] For state [1 1 0 1 1] :", val_2,
              env.background_color_monitor.background_ENDC)
        print(
            env.background_color_monitor.background_OKGREEN + "------------------------------------------------------------------------------------------------------------------------------------\n" + env.background_color_monitor.background_ENDC)
        print(env.background_color_monitor.background_OKCYAN + "[*] End" + env.background_color_monitor.background_ENDC)
        print(env.background_color_monitor.background_BOLD + """
ア マン カンノツ ウンデルスタンヅ ゼ アルツ ヘ イス スツヂイング イフ ヘ オンリ ロウク フオル ゼ エンヅ 
レズルツ ヰザオウツ タキング ゼ チメ ト =デレヴエ デウペリ イント ゼ レゾニング オフ ゼ スツヂ。

ミャモト ムサシ""" + env.background_color_monitor.background_ENDC)
