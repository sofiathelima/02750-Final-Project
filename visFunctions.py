# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 18:13:27 2023

@author: Asus
"""
import json
import os
import re
import numpy as np
import imageio
		
import matplotlib.pyplot as plt
from PIL import Image

class ImageObject():

    def __init__(self, 
              save_path,
              test_indices,
              num_classes:int=12):

        #randomize colors
#         color_span_r = np.linspace(0, 1, num=num_classes)
#         np.random.shuffle(color_span_r)
#         color_span_g = np.linspace(0, 1, num=num_classes)
#         np.random.shuffle(color_span_g)
#         color_span_b = np.linspace(0, 1, num=num_classes)
#         np.random.shuffle(color_span_b)	
        color_span_r = [0.90909091, 0.09090909, 0., 0.45454545, 0.63636364, 0.72727273, 0.54545455, 
                        0.18181818, 0.81818182, 0.27272727, 0.36363636, 1.]
        color_span_g = [0.09090909090909091, 0.36363636363636365, 0.7272727272727273, 0.6363636363636364, 
                        0.18181818181818182, 0.5454545454545454, 0.9090909090909092, 0.8181818181818182, 
                        0.0, 1.0, 0.4545454545454546, 0.2727272727272727]
        color_span_b = [0.8181818181818182, 0.5454545454545454, 0.7272727272727273, 0.4545454545454546, 
                        0.9090909090909092, 0.2727272727272727, 0.18181818181818182, 0.0, 0.6363636363636364, 
                        0.36363636363636365, 1.0, 0.09090909090909091]

        self.color_list = []
        for n in range(num_classes):
            self.color_list.append((color_span_r[n], color_span_g[n], color_span_b[n]))

        #directory to save to
        self.save_path = save_path

        #initial indexes for test dataset
        self.test_indices = test_indices

    def track_index(self,
                 removed_index,num_poses):
        # print('before',self.test_indices)
        self.test_indices = np.delete(self.test_indices,range(removed_index*num_poses,(removed_index+1)*num_poses))
        # print('after',self.test_indices)


    def track_index_single(self,
                 removed_index):
        # print('before',self.test_indices)
        self.test_indices = list(self.test_indices)
        del self.test_indices[removed_index]
        # print('after',self.test_indices)

    def create_images(self,
                   data, 
                   predict_probs,
                   num_instances,num_poses,training_round,
                   learning_mode):

        """
        data: numpy array
            node positions in space
        predict_arr: numpy array
            array of (#test data, # nodes, # classes) prediction from model
        training_round: int
            training round for saving the images			 
        """


        preds = [predicted.argmax(dim=-1) for predicted in predict_probs]
        # print('len(preds) inside create_images', len(preds), '  list of', type(preds[0]), 'each with size', preds[0].shape)
        for n in range(predict_probs.shape[0]):

            # 			predict = np.argmax(predict_arr[n,:,:], axis=-1)
            
            # # !!!!!!!!!!!!!!!! Check below !!!!!!!!!!!!!!
            patient_num = self.test_indices[n]//num_poses
            pose_num = n%num_poses
            
            
            fixed_color_list = [self.color_list[int(i)] for i in preds[n]]

            data_instance = data[n].x.numpy()
            X = data_instance[:, 0]
            Y = data_instance[:, 1]
            Z = data_instance[:, 2]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X, Y, Z, color=fixed_color_list, s=2)
            ax.view_init(elev=80, azim=10, roll=100)
            plt.axis('off')
            plt.title(f'{learning_mode} Instance: {training_round}')
            plt.savefig(os.path.join(self.save_path, f'{patient_num}-{pose_num}_{training_round}_{learning_mode}.jpg'), bbox_inches='tight', dpi=300)
            plt.close()
    # 			plt.show()

    def create_images_single(self,
                   data, 
                   predict_probs,
                   num_instances, 
				   training_round,
                   learning_mode):

        """
        data: numpy array
            node positions in space
        predict_arr: numpy array
            array of (#test data, # nodes, # classes) prediction from model
        training_round: int
            training round for saving the images			 
        """


        preds = predict_probs.argmax(axis=-1)
        # print('len(preds) inside create_images', len(preds), '  list of', type(preds[0]), 'each with size', preds[0].shape)

        for n in range(len(preds)):

            # 			predict = np.argmax(predict_arr[n,:,:], axis=-1)
            
            # # !!!!!!!!!!!!!!!! Check below !!!!!!!!!!!!!!
            patient_num = self.test_indices[n]
            pose_num = 0
            
            
            fixed_color_list = [self.color_list[int(i)] for i in preds[n]]

            data_instance = data[n].x.numpy()
            X = data_instance[:, 0]
            Y = data_instance[:, 1]
            Z = data_instance[:, 2]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X, Y, Z, color=fixed_color_list, s=2)
            ax.view_init(elev=80, azim=10, roll=100)
            plt.axis('off')
            plt.title(f'{learning_mode} Instance: {training_round}')
            plt.savefig(os.path.join(self.save_path, f'{patient_num}-{pose_num}_{training_round}_{learning_mode}.jpg'), bbox_inches='tight', dpi=300)
            plt.close()
    # 			plt.show()


    def create_GIF(self,
               picked_instance,
               picked_pose,
               learning_mode,
               frame_speed:int=4,
               movie_name:str='movie.gif'):

        """
        picked_instance: int
            final picked pose for GIF	 
        """
        
        filenames = os.listdir(self.save_path)
        print(self.save_path)
        gif_save_path = os.path.join(self.save_path, movie_name)
        with imageio.get_writer(gif_save_path, mode='I', fps=frame_speed) as writer:
            for img in filenames:
#                 print(re.match(f'{picked_instance}-{picked_pose}', img))
#                 if img.startswith(f'{picked_instance}-{picked_pose}'):
#                     if img
                if re.match(f'{picked_instance}-{picked_pose}', img):
#                     print(f'_{learning_mode}')
#                     print(img)
#                     print(re.match(f'_{learning_mode}', img))
#                     print(re.match(learning_mode, img) == learning_mode)
                    if img.split('.')[0].split('_')[-1] == learning_mode:
#                         print('writing')
                        image = imageio.imread(os.path.join(self.save_path, img))
                        writer.append_data(image)
#                     if re.match(str(learning_mode), img) == str(learning_mode):
#                         print('writing')
#                         image = imageio.imread(os.path.join(self.save_path, img))
#                         writer.append_data(image)


    def create_GIF_multiple(self,
               picked_instance,
               picked_pose,
               frame_speed:int=1,
               movie_name:str='movie_multi.gif'):

        """
        picked_instance: int
            final picked pose for GIF	 
        """
        
        filenames = os.listdir(self.save_path)
        learning_modes = ['Passive', 'MC', 'QBC']
        print(self.save_path)
        gif_save_path = os.path.join(self.save_path, movie_name)
#         fig, ax = plt.subplots(nrows=1, ncols=len(learning_modes))
		
        num_instances = 2
        for m in range(num_instances):
            fig, ax = plt.subplots(nrows=1, ncols=len(learning_modes), figsize=(8,3))
            for i, n in enumerate(learning_modes):
                for img in filenames:
                    if re.match(f'{picked_instance}-{picked_pose}_{m}_{n}', img):

                        plot_image = np.asarray(Image.open(os.path.join(self.save_path, img)))
                        ax[i].imshow(plot_image)
                        ax[i].axis('off')
						

            plt.savefig(os.path.join(self.save_path, f'multi_img_{picked_instance}-{picked_pose}_{m}.jpg'))
            plt.show()
				
        with imageio.get_writer(gif_save_path, mode='I', fps=frame_speed) as writer:
            for img in filenames:
#                 print(re.match(f'{picked_instance}-{picked_pose}', img))
#                 if img.startswith(f'{picked_instance}-{picked_pose}'):
#                     if img
                if re.match(f'multi_img_{picked_instance}-{picked_pose}', img):
                        image = imageio.imread(os.path.join(self.save_path, img))
                        writer.append_data(image)
#                     if re.match(str(learning_mode), img) == str(learning_mode):
#                         print('writing')
#                         image = imageio.imread(os.path.join(self.save_path, img))
#                         writer.append_data(image)

#Notes for 3D rotation
#for pose 1: 
# 	elev=80, azim=10, roll=100


# #for testing
# with open(f'./data_dicts/data_1.json', 'r') as f:
# 	graph_dict = json.load(f)
	
# x = graph_dict['x']
# x = np.array(x)

# save_path = r'./save_gif9'

# imgobj = ImageObject(save_path, 0)
# imgobj.create_GIF_multiple(4, 2)

# for n in range(3):
# 	predict_arr = np.random.choice(range(12), size=(10, 6890, 1))
# 	imgobj.create_images(x, predict_arr, n)
# imgobj.create_GIF(0)