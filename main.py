from AxelrodModel import AxelrodModel

model = AxelrodModel(N=64, width=8, height=8, features=2, traits=2)

for i in range(400000):
    print(i)
    model.step()

data = model.datacollector.get_model_vars_dataframe()
data.plot()
