import pkgutil

def load_plugins():
    plugins_prefix = 'axolotl.plugins.'
    for finder, name, ispkg in pkgutil.iter_modules():
        if name.startswith(plugins_prefix):
            module_name = name[len(plugins_prefix):]
            __import__(name)
            print(f"Loaded plugin: {module_name}")

load_plugins()
