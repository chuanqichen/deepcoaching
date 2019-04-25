# Sports Coaching from Pose Estimation

Problem

Good form is essential to success and safety in sports, and a role of a coach is to help athletes perfect their form. We hope to aid this process using visual recognition techniques. Specifically, we want to see if pose estimation state of the art is accurate enough for this task. By building and utilizing a pose estimation model, we can compare an amateur’s technique to that of a pro, and provide suggestions for improvement. We have golf in mind, since the swing can easily be broken down into a series of poses, but we hope our methods are generic. 

Related Work

Pose estimation is an active research area with many promising recent papers. We are interested in DensePose [1], which uses a variant of Mask-RCNN, work by Sun et al. [2] using high resolution networks, and more broadly, Xiao et al. [5] who provide a good comparison between models. Zecha et al [6] explored methods of improving pose detection specifically for sports. An additional area of interest is image generation using pose data [3], because this could be useful in the coaching context. If we have an amatuer in an incorrect pose, it would be helpful to generate a new image of them with the “corrected” technique. 

Methods and Algorithms

Refer to above figure, human poses’  keypoints location are modeled as one-hot masks and we adopt Mask-RCNN to predict masks, one for each of these keypoint types. From human poses’ key-points we can generate the measurement of kinematic coefficients, which are essential for continuous qualitative performance assessment. Posture visualization will be great for quantitative training feedback. If scope allows, we will use GANs [3] to generate images with a target pose. Most of the models we have considered have existing implementations in PyTorch, which we will utilize.

Dataset

We plan to use the COCO and MPII datasets, which have hundreds of thousands of labelled human poses. MPII specifically includes sports data such as golf. There is also the Leeds Sports Pose and Leeds Sports Pose Extended datasets with 2,000 and 10,000 images respectively. We plan to transfer learn on the broader datasets, not just sports poses. Once we have trained a sports pose estimation model we are confident in, we plan to apply it to curated set of pro and amataur golfers images gathered online so that we can compare the groups. 

Results Verification

For our pose estimation model, we plan to use the standard evaluation metric of Object Keypoint Similarity, which measures the distance between the detected keypoint and ground truth while taking into account factors such as visibility of the ground truth, object scale, etc. There are other potential quantitative metrics we can use, such as the closely related Geodesic Point Similarity, and the Ratio of Correct Point correspondences. We expect better athletes to have more consistent poses, and amateurs to have more variation. We are also interested in evaluating how the model performs on sports tasks compared to other activities. 

We will do qualitative analysis of our results by looking into particular instances of wrong predictions from the model. Examining those would inform us where the model is failing and also allow us to use sports domain knowledge to subjectively evaluate the pose against the model predictions. Another interesting qualitative analysis of the results will be to compare the predictions of the model for pro athletes versus amateurs. This again will give us a sense of model performance, as well as provide us an opportunity to use our judgment to draw interesting insights on the results.

Reference 

DensePose: Dense Human Pose Estimation In The Wild, Guler et al. https://arxiv.org/pdf/1802.00434.pdf
Deep High-Resolution Representation Learning for Human Pose Estimation, Sun et al. https://arxiv.org/pdf/1902.09212v1.pdf
Pose Guided Person Image Generation, Ma et al. https://arxiv.org/pdf/1705.09368.pdf
Mask R-CNN, He et al. https://arxiv.org/pdf/1703.06870.pdf
Simple Baselines for Human Pose Estimation and Tracking, Xiao et al. http://openaccess.thecvf.com/content_ECCV_2018/papers/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.pdf
Kinematic Pose Rectification for Performance Analysis and Retrieval in Sports, Zecha et al. http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w34/Zecha_Kinematic_Pose_Rectification_CVPR_2018_paper.pdf
