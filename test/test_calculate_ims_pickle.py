import pickle
import calculate_ims
import os.path


test_data_save_dir = 'pickled_test_data'
REALISATION = 'Hossack_HYP01-10_S1244'

def test_convert_str_comp():
    function = 'convert_str_comp'
    with open(os.path.join(test_data_save_dir, function + '_comp.P'), 'rb') as load_file:
        comp = pickle.load(load_file)
    value_to_test = calculate_ims.convert_str_comp(comp)
    with open(os.path.join(test_data_save_dir, function + '_converted_comp.P'), 'rb') as load_file:
        converted_comp = pickle.load(load_file)
    assert value_to_test == converted_comp

