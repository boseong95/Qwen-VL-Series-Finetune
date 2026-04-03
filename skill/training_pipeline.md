Job: make maintainable dataloading pipeline
You will use the this dataset root: /home/ubuntu/datasets/vlm

You will make a lot of llava format generators for each dataset.

Each input item can include either video or just a single image, or no image at all.

Alright then. 
Generate the json for the dataset and mix the dataset with coco vqa + Cholec80-VQA + SurgicalGPT-Cholec80-VQA. 
keep the llava.json generation script here /home/ubuntu/Qwen-VL-Series-Finetune/surgical_vqa_data

If things are okay, commit them to github. I guess you'd better commit ASAP as I am not 100% this instance keeps alive (maybe deepspeed problem..?)

Also, split the validation samples to consider all the different tasks (q1~qn). I guess total 50 inference is enough including all the tasks. (validation is enough only with vtrb_suturing dataset) 

For Cholec80-VQA + SurgicalGPT-Cholec80-VQA, you need to generate llava json also. (original video is here /home/ubuntu/datasets/vlm/Cholec80/videos)
I guess we can first try 
vtrb suturing : 50 %
coco: 25 %
others: 25 %
But we can keep trying to find the best mix to get the best validation result by trying only several training steps (less than ~1hr for each shot). Pick the best and go training.  
I think you can try more than batch > 1 (kill the current tmux session. It is running with batch = 1). 
Same lora configuration. make this kind of dataset mixing is configurable. 
make sure you save ckpt with lora part only.
