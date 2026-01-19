#from convcnp_assim_nz.utils.models.model_diagnostics import model_parameters_by_layer, model_dimensions_by_layer

def find_model_structure(model):
    """
    Function to print out the structure of a given model.
    Args:
        model: The model whose structure is to be printed.
    """
    for i, layer in enumerate(model.children()):
        print(f"Layer {i}: {type(layer)}")
        print(layer)
        print("--------------------")

def count_model_parameters(model):
    """
    Function to count the number of parameters in a given model.
    Args:
        model: The model whose parameters are to be counted.
    Returns:
        total_params: Total number of parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    return total_params


def model_parameters_by_layer(model, indent=0):
    """
    Find the number of parameters of each layer in the model.
    """
    if len(list(model.children())) == 0:
        layer_params = sum(p.numel() for p in model.parameters())
        print(f"{' ' * indent}Layer ({type(model)}): {layer_params} parameters")
    else:
        for i, layer in enumerate(model.children()):
            print(f"{' ' * indent}Layer {i} ({type(layer)}):")
            model_parameters_by_layer(layer, indent=indent+4)


def model_dimensions_by_layer(model, indent=0):
    """
    Find the input/output dimensions of each layer in the model.
    """

    if len(list(model.children())) == 0:
        print(f"{' ' * indent}Layer ({type(model)}):")
        for name, param in model.named_parameters():
            print(f"{' ' * (indent + 4)}Parameter: {name}, Shape: {param.shape}")
    else:
        for i, layer in enumerate(model.children()):
            print(f"{' ' * indent}Layer {i} ({type(layer)}):")
            model_dimensions_by_layer(layer, indent=indent+4)