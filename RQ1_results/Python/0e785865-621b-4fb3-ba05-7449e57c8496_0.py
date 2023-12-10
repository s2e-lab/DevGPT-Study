import pkg_resources
import pluggy

# Initialize Pluggy plugin manager
pm = pluggy.PluginManager("my_project")

def register_plugins(package_name, entry_group):
    try:
        distribution = pkg_resources.get_distribution(package_name)
        entry_map = distribution.get_entry_map()

        if entry_group in entry_map:
            for plugin_name, entry_point in entry_map[entry_group].items():
                # Load the module
                mod = entry_point.load()
                
                # Register the module with Pluggy
                pm.register(mod, plugin_name)
                
                print(f"Registered plugin: {plugin_name}")

    except pkg_resources.DistributionNotFound:
        print(f"{package_name} is not installed.")
        
# Replace 'your-package-name' and 'your-entry-group' with the relevant values
package_name = "datasette-write-ui"
entry_group = "datasette"

register_plugins(package_name, entry_group)
