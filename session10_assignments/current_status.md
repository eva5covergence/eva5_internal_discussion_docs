##Session10 assignment

**Updated files:**


basic_config.py
model_builder.py
misc_utils.py
base_data_transforms.py

**New files:**

get_optimizer.py
get_lr_scheduler.py
lr_finder.py
resnet_session9_lr_finder.py

Session10_assignment_lr_finder.ipynb
Session10_assignment.ipynb

**Implentation details:**

- Implemented reduceLRplataeu
- Implemented rotate and cutout
- Modularised optimiser and lr_scheduler related code
- lr_finder for sgd
- train_acc and train_loss curve - just updated main notebook by passing the train_acc and train_loss details to plot the graphs
- Max test acc got 89.29 % 

**Pending items**

- display 25 misclassified images of grad cam - make it look as gallery of images (image grid)

**lr_finder results at multiple runs for lr_finder.get_best_lr_sgd(model, train_loader, lr=1e-7, momentum=0.9, end_lr=100, num_iter=100)**

1.52E-01
1.00E-01
6.58E-02
1.00E-01
1.00E-01



