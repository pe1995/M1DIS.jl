"""
Configuration loader for M1DIS and M3DIS projects.

This module provides utilities to load eos_config.toml and properly expand
relative paths from a base directory to absolute paths.
"""
module ConfigLoader

using TOML

"""
    load_config(config_path::String, base_dir::String = @__DIR__)

Load eos_config.toml and expand all relative paths to absolute paths.

The base_dir should point to where the eos_config.toml is located or can be found.
All relative paths in the config are resolved relative to base_dir.

# Arguments
- `config_path`: Path to the TOML config file (relative or absolute)
- `base_dir`: Base directory for resolving relative paths

# Returns
- `config_dict`: Dictionary with all paths expanded to absolute paths
"""
function load_config(config_path::String, base_dir::String = @__DIR__)
    # Resolve config file path
    config_file = if isabspath(config_path)
        config_path
    else
        path = joinpath(base_dir, config_path)
        isfile(path) ? path : joinpath(base_dir, "configs", config_path)
    end
    
    if !isfile(config_file)
        error("Configuration file not found: $config_file")
    end
    
    # Load TOML
    config = TOML.parsefile(config_file)
    
    # Expand paths to absolute paths
    _expand_paths!(config, base_dir)
    
    return config
end

"""
    expand_path(path::String, base_dir::String)

Expand a path (relative or absolute) to an absolute path.

If path is already absolute, return as-is.
If path is relative, resolve it relative to base_dir.

# Arguments
- `path`: The path to expand (can be empty string)
- `base_dir`: The base directory for resolving relative paths

# Returns
- Absolute path string
"""
function expand_path(path::String, base_dir::String)
    isempty(path) && return path
    
    if isabspath(path)
        return normpath(path)
    else
        expanded = normpath(joinpath(base_dir, path))
        return expanded
    end
end

"""
    _expand_paths!(config::Dict, base_dir::String)

Recursively expand all path-like keys in the config dictionary to absolute paths.

Only expands keys for m3d_dir and dispatch_dir, as other paths should be used
within @in_dispatch and @in_tumult macros which handle relative paths correctly.
"""
function _expand_paths!(config::Dict, base_dir::String)
    path_keys = [
        "m3d_dir", "dispatch_dir", "marcs_dir"
    ]
    
    for key in path_keys
        if haskey(config, key) && !isempty(config[key])
            config[key] = expand_path(config[key], base_dir)
        end
    end
    
    # Recursively process nested dicts
    for (key, value) in config
        if isa(value, Dict)
            _expand_paths!(value, base_dir)
        end
    end
    
    return config
end

end # module ConfigLoader
