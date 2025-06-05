# This will be a simple function to extract information from a user's qiskit-json file

import json, os
from qiskit_ibm_runtime import QiskitRuntimeService

def get_creds(args):
    """This function determines the user's IBM Quantum channel, instance, and token, using values provided
    within the config.yaml file or as defined within the user's qiskit configuration from provided qiskit_json_path
    specified in the config.yaml file, and then parses its contents. It returns the main items in this json file,
    such as the instance and api token, which can then be passed into the QML functions when using a real
    hardware backend.
    
    Args: 
        args: arguments passed in from the config.yaml file.
        
    Returns:
        dictionary that can be used to instantiate a QiskitRuntimeService
    """
    cred_source_dict = {'channel':'ibm_channel', 'instance':'ibm_instance', 'token':'ibm_token', 'url':'ibm_url'}
    rval = {}
    for ibm_name, yaml_name in cred_source_dict.items():
        value = args.get(yaml_name, None)
        if value:
            rval[ibm_name] = value

    qiskit_json_path = args.get('qiskit_json_path', None)
    if qiskit_json_path:
        qiskit_json_path = os.path.expanduser(qiskit_json_path)
        if os.path.exists(qiskit_json_path):
            # load the qiskit json file
            with open(qiskit_json_path, 'r') as jfile:
                creds = json.load(jfile)
            # Access keys and values
            # The items we want are actually in a nested dictionary, so we have to loop through the outer dictionary first, then the 
            # nested one.  This nested dictionary (outer_value) is actually the value for the key in the parent dictionary. 
            for outer_key, outer_value in creds.items():
                for ibm_name in cred_source_dict.keys():
                    if ibm_name not in rval:
                        value = outer_value.get(ibm_name, None)
                        if value:
                            rval[ibm_name] = value
        else:
            print('IBM credentials not found! Please verify that the path to your qiskit-ibm.json file is correct.')
    return rval

def instantiate_runtime_service(args):
    """Quick way to instantiate QiskitRuntimeService in one place. A basic call to this function can then be done in anywhere else."""
    return QiskitRuntimeService(**get_creds(args))
