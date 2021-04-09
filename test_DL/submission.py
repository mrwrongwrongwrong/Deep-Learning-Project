# Feel free to modifiy this file.

from torchvision import models, transforms


team_id = 10
team_name = "dl10"
email_address = "team_leader_nyu_email_address@nyu.edu"

def get_model():
	#model = models.resnet50(num_classes=800)
	#output = model(unlabeled_data)
	#another_data=zip(unlabel_data,output)
	#model2 = model().train(another_data)
	#return model2;

    return models.resnet50(num_classes=800)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])