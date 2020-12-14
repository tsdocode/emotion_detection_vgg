from emotion.model.vgg19 import vgg19


ctg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg = vgg19(3, 7, False)

print(vgg)