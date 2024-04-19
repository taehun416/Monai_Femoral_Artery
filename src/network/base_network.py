import logging

class ModelRegistry:
    """
    Registry for dynamic creation and management of models.
    """
    _registry = {}

    @classmethod
    def register(cls, model_name):
        """
        Registers a model class under a specified name.

        Args:
            model_name (str): The name for the model class registration.

        Returns:
            function: A decorator function for model class registration.
        """
        def decorator(model_class):
            """
            Decorator function that registers the model class.

            Args:
                model_class (class): The model class to be registered.

            Returns:
                class: The registered model class.
            """
            if model_name in cls._registry:
                logging.warning(f"Warning: Model '{model_name}' is already registered. Overwriting.")
            cls._registry[model_name] = model_class
            return model_class

        return decorator
    
    @classmethod
    def create(cls, model_name, *args, **kwargs):
        """
        Creates an instance of a registered model class by name.

        Args:
            model_name (str): The name of the model class to instantiate.
            *args: Variable length argument list for the model class constructor.
            **kwargs: Arbitrary keyword arguments for the model class constructor.

        Returns:
            instance: An instance of the requested model class.

        Raises:
            KeyError: If the specified model name is not found in the registry.
        """
        if model_name not in cls._registry:
            raise KeyError(f"Model '{model_name}' not found in registry.")
        return cls._registry[model_name](*args, **kwargs)
