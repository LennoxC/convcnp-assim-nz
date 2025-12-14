import torch
import numpy as np



def copy_unet_weights_except_first(model, old_model):
    """
    This is to copy the weights from an old model's UNet to a new model's UNet,
    except for the first layer in before_turn_layers which often has different input channels in a 
    transfer learning scenario. I.e. more input channels in the new model than the old model.

    Don't accidentally pass old_model, model into this function otherwise you'll be 
    debugging for an hour like I did, and get really confused. Pass model, old_model.

    Returns the new model with the copied weights, with the first layer weights randomly initialized.

    # TODO: add an initialization strategy for the first layer weights
    """
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
    """
    Freeze all UNet layers except the first before_turn_layer.
    Useful for transfer learning scenarios, when you first want to train the randomly initialized
    first layer, before unfreezing the rest of the UNet for fine-tuning.
    """
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