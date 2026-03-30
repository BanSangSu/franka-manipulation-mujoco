   # Intelligent Robotic Manipulation
   
   The overall Goal of the project is to grasp a YCB-Object and place it in a goal basket while avoiding obstacles. To do this we have provided some helpful tips/information on various subtasks that you may or may not need to solve. Perception (task 1) to detect the graspable object, Controller (task 2) to move your robot arm, sample and execute a grasp (task 3), localize and track obstacles (task 4) and plan the trajectory to place the object in the goal, while avoiding the obstacles (task 5). Note that, what we mention in the task subparts are just for guidance and you are fully free to choose whatever you want to use to accomplish the full task. But you need to make sure that you don't use privilege information from the sim in the process.
   
   ### Things you can change and data that you can use:
   
   - Testing with custom objects
   - Testing with different control modes
   - Toggling obstacles on and off
   - Adding new metrics for better report and explanation
   - The `user_camera` position and orientation
   - Information such as:
       - Robot joint information
       - Robot gripper and end-effector information
       - Any information from camera (RGB and Depth)
       - Goal receptacle position
       - Camera position/matrices
       - Contact information (but not for querying objects)
  
   Note: You are not allowed to use any information from the sim that can skip solving the tasks, like ground truth object positions, obstacle positions, re-configuring robot, etc.
   
   ### Things you cannot change without an explicit request to course TA's/Professor:
   
   - Any random sampling in the sim
   - Number of obstacles
   - Ground truth position of the object in question
   - Any ground truth orientation
   - Robot initial position and arm orientation
   - Goal receptacle position
   
   *Note: If you want to add a new metric you can use ground truth information there but only for comparison with your prediction.*
   
   Also you can test out your system with different configurations but the final method should only be tested on the given configuration.
   
   ### Checkpoints & Marks:
   
   - Code Related
       - Being able to detect the object and get it’s pose (+10)
       - Moving the arm to the object (+10)
       - Being able to grasp object (+15)
       - Being able to move the arm with the object to goal position (without obstacles) (+20)
       - Detecting and tracking the obstacles (+10)
       - Full setup: Being able to execute pick and place with obstacles present (+25)
   - A part of your marks is also fixed on the report (+10)
   
   - You need to complete at all the given **5** tasks to qualify for the full marks.

   
   - We will only consider the checkpoints as complete if you provide a metric or a success rate for each.
   - The format of the report will be the standard TU-Darmstadt format.
   - You can choose any 6 objects from the available 10 objects to show your results.
   - Please report relevant metrics or success rates by doing 10 different evaluations with the same object.
   - Your code should be runnable with a single script and experiment configuration.
   
   ### Submission format:
   
   - Link to your github/gitlab repository containing a well documented README with scripts to run and test various parts of the system.
   - Report PDF.
   
   ## Task 1 (Perception)
   
   You can use a 6D object pose estimation workflow using a simulation environment.
   
   *Reference*
   Global registration (coarse pose estimation) [Tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
   ICP registration (coarse pose estimation) [Tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
   MegaPose, a method to estimate the 6D pose of novel objects, that is, objects unseen during training. [repo](https://github.com/megapose6d/megapose6d)
   
   ## Task 2 (Control)
   
   To control the robot an IK-solver can be used. You can use the pseudo-inverse or the transpose based solution. Use Your IK-solver to move the robot to a certain goal position. This Controller gets used throughout the project (e.g. executing the grasp - moving the object to the goal).
   
   ## Task 3 (Grasping)
   
   Now that you have implemented a controller (with your IK solver) and tested it properly, it is time to put that to good use. From picking up objects to placing them a good well-placed grasp is essential. Hence, given an object you have to design a system that can effectively grasp it. You can use the model from the [Grasping exercise](https://github.com/iROSA-lab/GIGA)
   and [colab](https://colab.research.google.com/drive/1P80GRK0uQkFgDbHzLjwahyJOalW4M5vU?usp=sharing) to sample a grasp from a point-cloud. We have added a camera, where you can specify its position. You can set the YCB object to a fixed one (e.g. a Banana) for development. Showcase your ability to grasp random objects
   for the final submission.
   
   ## Task 4 (Localization & Tracking)
   
   After you have grasped the object you want to place it in the goal-basket. In order to avoid the obstacles (red and orange spheres), you need to track them. Use the provided fixed camera and your custom-positioned cameras as sensors to locate and track the obstacles. Visualize your tracking capabilities in the Report (optional) and use this information to avoid collision with them in the last task. You could use a Kalman Filter.
   
   ## Task 5 (Planning)
   
   After you have grasped the YCB object and localized the obstacle, the final task is to plan the robot’s movement in order to place the object in the goal basket. Implement a motion planner to avoid static and dynamic obstacles and execute it with your controller. Once you are above the goal-basket open the gripper to drop the object in the goal.
   
   Motion planning can be generally decoupled into global planning and motion planning. Global planning is responsible for generate a reference path to avoid static obstacles while local planning is for keeping track of the reference path and avoid dynamic obstacles.
   
   - For global planning,
       - An easy way is to use sampling-based methods (rrt, prm) to sample the reference path directly in 3d space and use your designed IK solver to track the path, see [here](https://github.com/yijiangh/pybullet_planning/tree/dev/src/pybullet_planning/motion_planners) for more algorithmic details.
       - A faster but difficult solution is to sample the path directly in the configuration space so here you do not need the IK solver. You can see [here](https://github.com/sea-bass/pyroboplan) for an example, though it is not implemented using pybullet.
   - For local planning, after you have a prediction of the moving obstacles,
       - An easy way is to use potential field method to avoid them. You can check [here](https://github.com/PulkitRustagi/Potential-Field-Path-Planning) for more details.
       - A more advanced approach is to use MPC or sampling-based MPC to handle the moving obstacles. You can check [this](https://github.com/tud-amr/m3p2i-aip) for more details, though the kinematic model and collision checking are done in IsaacGym.
   
   You can decide to use whichever techniques to solve the task, you can use either a `global planner` and a `local planner` combination, or you can directly use the sampling-based MPC to avoid static and dynamic obstacles.
   
 
 Note: We have only shown one of the many possible pathways to solve the task. Feel free to explore and experiment with different approaches.
# Final Words:
 We hope you have fun and explore robotics more deeply through this project.
