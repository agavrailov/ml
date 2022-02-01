from ludwig.api import LudwigModel
model = LudwigModel.load("/home/user/Documents/ludwig-test/plus1/results/api_experiment_run_0/model")
x = model.predict({"numberIn":[1]}, return_type='dict')
#x = model.predict({"numberIn":[1]}, return_type=<class 'dict'>) I tried this with no success
print(x)