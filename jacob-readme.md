Hello and welcome to my readme. This is a fork of the Stanford AC-Teach for testing the Discrim-RL framework. I focused my research on the pick-and-place setting. Follow the setup from the Stanford page. Easiest thing to do is make a virtual environment -- make sure you get the right version of gym and tensorflow; the requirements file might be different from the published version (it was when I was trying to get setup and was very frustrating). 


Random bits:
 - there's a graphics/rendering/screen bug that i dont fully understand. if you are remote and don't want to render to screen, you need to run 'unset LD_PRELOAD'. 
 - when you make a new gym env / change an existing gym env file, you have to recreate the build by running 'pip install -e'


Research things:
 - you can run their scripts by following the directions on their page
 - /models contains /rl and /supervised. /rl has simple_actor.py which is the base actor model as well as learned_value.py which has an output for the learned value as well. /supervised has recurrent_frame_discriminator.py, which is the baseline discriminator model.
 - /scripts has all the fun things! 
	- run.py, test_using_stable.py are their files
	- experts_control.py is a testing file for the experts. It has the most simple example of interacting with the env and experts. 
	- text_experts.py has the functions i used to try new things with the experts like doubling the length of the expert rollout and other things.
	- train_sample.py is just for simple RL, no discriminator
	- make_video.py takes a folder of an episode and makes a video
	- training things! 
		- train_recurrent_discrim.py is the simplest connection of the rl platform with the basic discriminaotr
		- train_INTERPOLATED_recurrent_discrim.py extends the length of the expert episode out to that of the actor's episode.
		- train_ANNEALED_recurrent_discrim.py anneals the discrim-based reward with the environment reward

run things in an activated environment with 'python scripts/train_whatever_you_want.py 
