To use Q-learning algorithm to train an agent to traverse and navigate in Minecraft, there are several prerequisites and tools needed. First, the user needs to have a Minecraft account and download Minecraft version 1.15.2. Additionally, the user needs to start the Minecraft server using the following commands:

Navigate to the directory containing the Minecraft Spigot server files: cd minecraft_spigot-1.15.2
Start the server: java -Xmx1024M -Xms1024M -jar spigot-1.15.2.jar
This enables the mcpi package to interact with the Minecraft agent.

To set up the environment for training, the user needs to launch Minecraft, click on the "Multiplayer" option, click on "Add Server", and enter "localhost" in the server address field. Next, resize the Minecraft window to the maximum size, but do not fullscreen it. It is important to press the F1 key to remove the hotbar and make the vision horizontal.

There are three Python files needed to train the agent: HistCorr.py, Minecraft_Manipulate.py, and main.py.

HistCorr.py handles the images, performs image comparison, and returns a correlation number. This correlation number is later used in selecting states for training the agent.

Minecraft_Manipulate.py contains many basic functions that can manipulate the Minecraft environment, such as moving the agent, turning, initializing direction, taking screenshots, resizing images, and calculating move speed.

Main.py contains the implementation of the Q-learning algorithm for training the agent and the navigation function for testing the trained agent. It includes getting state, reward, action, updating image lists, storing learning results, loading learning results, and other functions.

To use the Q-learning algorithm to train the agent, the user needs to create a folder inside the training result folder with four sub-folders named "left_block_list", "right_block_list", "left_pre_list", and "right_pre_list". These folders will store the learning results. To train the agent, the user needs to call the Q-learning function and then the navigation function. Both functions will include a time.sleep() function to pause the execution for 5 seconds, allowing the user to switch back to the Minecraft window and wait for the agent to complete the action.


Inside the main.py file, there are several global variables that can be modified to alter the behavior of the Q-learning algorithm and the Minecraft agent. These variables include the exploration rate, learning rate, discount factor, number of episodes to train for and etc. Comments are comprehensive inside the codes.

The following packages were installed for the project: mcpi, pynput, and other primary packages were installed using pip
pip install mcpi
pip install pynput
