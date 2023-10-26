using Metalhead: VGG19, preprocess, load, labels
# using the packages that we have added
using Flux: onecold
# using flux package
model = VGG19()
class_labels = labels(model)

println("Enter the name of your file: ")
user_image = preprocess(load(readline()))
model_prediction = model(user_image)
top_class = onecold(model_prediction)[1]
class_name = labels[top_class]

println("I think this image contains: $(class_name)")