import torch
import numpy as np

# TODO: add an initialization strategy for the first layer weights

"""
Before using this function, model looks like this:
UNet(
  (before_turn_layers): ModuleList(
    (0): Conv2d(14, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1-4): 4 x Sequential(
      (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (1): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
  )
  (after_turn_layers): ModuleList(
    (0): Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1-3): 3 x Sequential(
      (0): Upsample(scale_factor=2.0, mode='bilinear')
      (1): Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (4): Sequential(
      (0): Upsample(scale_factor=2.0, mode='bilinear')
      (1): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
  )
  (final_linear): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
)

After the function, I get this which doesn't match:
UNet(
  (before_turn_layers): ModuleList(
    (0): Conv2d(11, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1-4): 4 x Sequential(
      (0): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (1): AvgPool2d(kernel_size=2, stride=2, padding=0)
    )
  )
  (after_turn_layers): ModuleList(
    (0): Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1-3): 3 x Sequential(
      (0): Upsample(scale_factor=2.0, mode='bilinear')
      (1): Conv2d(128, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
    (4): Sequential(
      (0): Upsample(scale_factor=2.0, mode='bilinear')
      (1): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    )
  )
  (final_linear): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
)
"""

def copy_unet_weights_except_first(model, old_model):
    
    old_unet = old_model.model.decoder.links[0]
    unet = model.model.decoder.links[0]

    # copy before_turn_layers weights
    for old_layer, new_layer in zip(old_unet.before_turn_layers, unet.before_turn_layers):
        if (isinstance(old_layer, torch.nn.modules.container.Sequential)):
            new_layer[0].load_state_dict(old_layer[0].state_dict())

    # copy after_turn_layers weights
    for old_layer, new_layer in zip(old_unet.after_turn_layers, unet.after_turn_layers):
        if (isinstance(old_layer, torch.nn.modules.conv.Conv2d)):
            new_layer.load_state_dict(old_layer.state_dict())
        
        if (isinstance(old_layer, torch.nn.modules.container.Sequential)):
            new_layer.load_state_dict(old_layer.state_dict())

    # copy the final layer weights
    unet.final_linear.load_state_dict(old_unet.final_linear.state_dict())

    # verify weights copied correctly on some sample layers
    assert np.allclose(old_unet.before_turn_layers[1][0].weight.cpu().detach().numpy(), unet.before_turn_layers[1][0].weight.cpu().detach().numpy())
    assert np.allclose(old_unet.after_turn_layers[1][1].weight.cpu().detach().numpy(), unet.after_turn_layers[1][1].weight.cpu().detach().numpy())
    assert np.allclose(old_unet.final_linear.weight.cpu().detach().numpy(), unet.final_linear.weight.cpu().detach().numpy())

    # return the model with the updated weights
    return model

# Freeze all unet layers except the first before_turn_layer
# unfreeze = False to freeze, True to unfreeze
def freeze_unet_except_first(model, unfreeze=False):
    unet = model.model.decoder.links[0]

    # freeze before_turn_layers except the first layer
    for i, layer in enumerate(unet.before_turn_layers):
        if i != 0:
            for param in layer.parameters():
                param.requires_grad = unfreeze

    # freeze after_turn_layers except the first layer
    for i, layer in enumerate(unet.after_turn_layers):
        for param in layer.parameters():
            param.requires_grad = unfreeze

    # freeze the final layer
    for param in unet.final_linear.parameters():
        param.requires_grad = unfreeze

    return model