import json

class Config:
    def update(self, cfg):
        """
        Update the configuration attributes based on the input dictionary or JSON string.
        :param cfg: A dictionary or JSON string containing the configuration updates.
        """
        if cfg is None:
            return
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)  # Convert JSON string to dictionary
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON string.")
        if not isinstance(cfg, dict):
            raise ValueError("Configuration must be a dictionary or JSON string.")

        for key, value in cfg.items():
            if hasattr(self, key):  # Update only existing attributes
                setattr(self, key, value)

    def get_dict(self):
        """
        Get all attributes of the class as a dictionary.
        """
        attributes = {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }
        return attributes  # Return as a dictionary


# class A_config(Config):
#     def __init__(self, cfg = {'a': 1, 'b': "0"}):
#         self.a = 0
#         self.b = "5"
#         # Update attributes if cfg is provided
#         if cfg:
#             self.update(cfg)