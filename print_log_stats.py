import os                                                                                                         
from tensorboard.backend.event_processing import event_accumulator                                                
run_dir = './logs/runs/test_ppodr_novel_cubes_only_descreet_0.1_no_norm'                                         
files = [f for f in os.listdir(run_dir) if f.startswith('events.out')]                                            
for f in files:
    print(f)
    ea = event_accumulator.EventAccumulator(os.path.join(run_dir, f))                                          
    ea.Reload()                                                                                                       
    metrics = ['discriminator_accuracy','target_discriminator_accuracy','reward_novelty_to_diayn_abs_ratio',
        'grasp_success_rate','descriptor_norm_mean','descriptor_feature_var_mean',
         'cube_positions_std','final_cube_z_mean']                                                                         
    for metric in metrics:                                                                                            
        if metric not in ea.Tags()['scalars']:                                                                        
            print(metric, 'not logged')                                                                               
            continue                                                                          
        events = ea.Scalars(metric)                                                           
        print(metric, 'len', len(events))                                                     
        if events:                                                                            
            print(' last', events[-1].value)                                                  
            print(' recent avg', sum(e.value for e in events[-10:])/len(events[-10:])) 
