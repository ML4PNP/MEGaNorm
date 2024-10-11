
import json
import os


def get_relative_path(filename):
    """ Function to get the path relative to the .py file

    Args:
        filename (str): _description_

    Returns:
        str: _description_
    """
    
    script_dir = os.path.dirname(__file__)  # Get the directory of the current script
    return os.path.join(script_dir, filename + '.json')


def save_sensor_layouts(layout, filename):
    """Function to save the sensor layouts to a JSON file

    Args:
        layout (dict): _description_
        filename (str): _description_
    """
    
    full_path = get_relative_path(filename)
    
    with open(full_path, 'w') as f:
        json.dump(layout, f, indent=4)
    print(f"Sensor layouts saved to {full_path}")
    

def add_specific_layout(device, layout, layout_data):
    """ Function to add a specific sensor layout to a JSON file
    
    Args:
        device (str): _description_
        layout (str): _description_
        layout_data (dict): _description_
 
    """
    
    full_path = get_relative_path(device)
    
    try:
        # Load the existing data
        with open(full_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, create a new dictionary
        data = {}

    # Add or update the specific layout
    data[layout] = layout_data

    # Save the updated data back to the file
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Layout '{layout}' saved to {full_path}")



def load_specific_layout(device, layout):
    """
    Function to load a specific sensor layout from a JSON file
    Args:
        device (str): _description_
        layout (str): _description_

    Returns:
        dict: _description_
    """
    
    full_path = get_relative_path(device)
    
    try:
        # Load the existing data
        with open(full_path, 'r') as f:
            data = json.load(f)
        # Return the specific layout if it exists
        if layout in data:
            return data[layout]
        else:
            print(f"Layout '{layout}' not found in {full_path}")
            return None
    except FileNotFoundError:
        print(f"File '{full_path}' not found")
        return None


def create_layouts(modality='MEG', device='MEGIN'):
    
    if modality=='MEG' and device=='MEGIN':
        MEGIN_layouts = {
            
            "Megin_MAG_All":{ 
                'MAG_All': [
                    "MEG0121", "MEG0341", "MEG0311", "MEG0321", "MEG0511", "MEG0541", "MEG0331", "MEG0521", "MEG0531", 
                    "MEG0611", "MEG0641", "MEG0621", "MEG0821", "MEG1411", "MEG1221", "MEG1211", "MEG1231", "MEG0921", 
                    "MEG0931", "MEG1241", "MEG0911", "MEG0941", "MEG1021", "MEG1031", "MEG0811", "MEG1011", "MEG0111", 
                    "MEG0131", "MEG0211", "MEG0221", "MEG0141", "MEG1511", "MEG0241", "MEG0231", "MEG1541", "MEG1521", 
                    "MEG1611", "MEG1621", "MEG1531", "MEG1421", "MEG1311", "MEG1321", "MEG1441", "MEG1431", "MEG1341", 
                    "MEG1331", "MEG2611", "MEG2411", "MEG2421", "MEG2641", "MEG2621", "MEG2631", "MEG0411", "MEG0421", 
                    "MEG0631", "MEG0441", "MEG0431", "MEG0711", "MEG1811", "MEG1821", "MEG0741", "MEG1631", "MEG1841", 
                    "MEG1831", "MEG2011", "MEG1041", "MEG1111", "MEG1121", "MEG0721", "MEG1141", "MEG1131", "MEG0731", 
                    "MEG2211", "MEG2221", "MEG2241", "MEG2231", "MEG2441", "MEG2021", "MEG1641", "MEG1721", "MEG1711", 
                    "MEG1911", "MEG1941", "MEG1731", "MEG2041", "MEG1921", "MEG1931", "MEG1741", "MEG2111", "MEG2141", 
                    "MEG2431", "MEG2521", "MEG2531", "MEG2311", "MEG2321", "MEG2511", "MEG2031", "MEG2341", "MEG2331", 
                    "MEG2541", "MEG2121", "MEG2131"],
            },
            
            "Megin_All_Lobe":{
                "MAG_frontal_left": ["MEG0121", "MEG0341", "MEG0311", "MEG0321", "MEG0511", "MEG0541", "MEG0331", "MEG0521", "MEG0531", "MEG0611", "MEG0641", "MEG0621", "MEG0821"], 
                "MAG_frontal_right": ["MEG1411", "MEG1221", "MEG1211", "MEG1231", "MEG0921", "MEG0931", "MEG1241", "MEG0911", "MEG0941", "MEG1021", "MEG1031", "MEG0811", "MEG1011"], 
                "MAG_temporal_left": ["MEG0111", "MEG0131", "MEG0211", "MEG0221", "MEG0141", "MEG1511", "MEG0241", "MEG0231", "MEG1541", "MEG1521", "MEG1611", "MEG1621", "MEG1531"], 
                "MAG_temporal_right": ["MEG1421", "MEG1311", "MEG1321", "MEG1441", "MEG1431", "MEG1341", "MEG1331", "MEG2611", "MEG2411", "MEG2421", "MEG2641", "MEG2621", "MEG2631"], 
                "MAG_parietal_left": ["MEG0411", "MEG0421", "MEG0631", "MEG0441", "MEG0431", "MEG0711", "MEG1811", "MEG1821", "MEG0741", "MEG1631", "MEG1841", "MEG1831", "MEG2011"], 
                "MAG_parietal_right": ["MEG1041", "MEG1111", "MEG1121", "MEG0721", "MEG1141", "MEG1131", "MEG0731", "MEG2211", "MEG2221", "MEG2241", "MEG2231", "MEG2441", "MEG2021"], 
                "MAG_occipital_left": ["MEG1641", "MEG1721", "MEG1711", "MEG1911", "MEG1941", "MEG1731", "MEG2041", "MEG1921", "MEG1931", "MEG1741", "MEG2111", "MEG2141"], 
                "MAG_occipital_right": ["MEG2431", "MEG2521", "MEG2531", "MEG2311", "MEG2321", "MEG2511", "MEG2031", "MEG2341", "MEG2331", "MEG2541", "MEG2121", "MEG2131"], 
                "GRAD1_frontal_left": ["MEG0122", "MEG0342", "MEG0312", "MEG0322", "MEG0512", "MEG0542", "MEG0332", "MEG0522", "MEG0532", "MEG0612", "MEG0642", "MEG0622", "MEG0822"], 
                "GRAD1_frontal_right": ["MEG1412", "MEG1222", "MEG1212", "MEG1232", "MEG0922", "MEG0932", "MEG1242", "MEG0912", "MEG0942", "MEG1022", "MEG1032", "MEG0812", "MEG1012"], 
                "GRAD1_temporal_left": ["MEG0112", "MEG0132", "MEG0212", "MEG0222", "MEG0142", "MEG1512", "MEG0242", "MEG0232", "MEG1542", "MEG1522", "MEG1612", "MEG1622", "MEG1532"], 
                "GRAD1_temporal_right": ["MEG1422", "MEG1312", "MEG1322", "MEG1442", "MEG1432", "MEG1342", "MEG1332", "MEG2612", "MEG2412", "MEG2422", "MEG2642", "MEG2622", "MEG2632"], 
                "GRAD1_parietal_left": ["MEG0412", "MEG0422", "MEG0632", "MEG0442", "MEG0432", "MEG0712", "MEG1812", "MEG1822", "MEG0742", "MEG1632", "MEG1842", "MEG1832", "MEG2012"], 
                "GRAD1_parietal_right": ["MEG1042", "MEG1112", "MEG1122", "MEG0722", "MEG1142", "MEG1132", "MEG0732", "MEG2212", "MEG2222", "MEG2242", "MEG2232", "MEG2442", "MEG2022"], 
                "GRAD1_occipital_left": ["MEG1642", "MEG1722", "MEG1712", "MEG1912", "MEG1942", "MEG1732", "MEG2042", "MEG1922", "MEG1932", "MEG1742", "MEG2112", "MEG2142"], 
                "GRAD1_occipital_right": ["MEG2432", "MEG2522", "MEG2532", "MEG2312", "MEG2322", "MEG2512", "MEG2032", "MEG2342", "MEG2332", "MEG2542", "MEG2122", "MEG2132"], 
                "GRAD2_frontal_left": ["MEG0123", "MEG0343", "MEG0313", "MEG0323", "MEG0513", "MEG0543", "MEG0333", "MEG0523", "MEG0533", "MEG0613", "MEG0643", "MEG0623", "MEG0823"], 
                "GRAD2_frontal_right": ["MEG1413", "MEG1223", "MEG1213", "MEG1233", "MEG0923", "MEG0933", "MEG1243", "MEG0913", "MEG0943", "MEG1023", "MEG1033", "MEG0813", "MEG1013"], 
                "GRAD2_temporal_left": ["MEG0113", "MEG0133", "MEG0213", "MEG0223", "MEG0143", "MEG1513", "MEG0243", "MEG0233", "MEG1543", "MEG1523", "MEG1613", "MEG1623", "MEG1533"], 
                "GRAD2_temporal_right": ["MEG1423", "MEG1313", "MEG1323", "MEG1443", "MEG1433", "MEG1343", "MEG1333", "MEG2613", "MEG2413", "MEG2423", "MEG2643", "MEG2623", "MEG2633"], 
                "GRAD2_parietal_left": ["MEG0413", "MEG0423", "MEG0633", "MEG0443", "MEG0433", "MEG0713", "MEG1813", "MEG1823", "MEG0743", "MEG1633", "MEG1843", "MEG1833", "MEG2013"], 
                "GRAD2_parietal_right": ["MEG1043", "MEG1113", "MEG1123", "MEG0723", "MEG1143", "MEG1133", "MEG0733", "MEG2213", "MEG2223", "MEG2243", "MEG2233", "MEG2443", "MEG2023"], 
                "GRAD2_occipital_left": ["MEG1643", "MEG1723", "MEG1713", "MEG1913", "MEG1943", "MEG1733", "MEG2043", "MEG1923", "MEG1933", "MEG1743", "MEG2113", "MEG2143"], 
                "GRAD2_occipital_right": ["MEG2433", "MEG2523", "MEG2533", "MEG2313", "MEG2323", "MEG2513", "MEG2033", "MEG2343", "MEG2333", "MEG2543", "MEG2123", "MEG2133"]
            },
            "Megin_GRAD_All":{ 
                'GRAD_All': ['MEG0122', 'MEG0342', 'MEG0312', 'MEG0322', 'MEG0512', 'MEG0542', 'MEG0332', 'MEG0522', 'MEG0532',
                            'MEG0612', 'MEG0642', 'MEG0622', 'MEG0822', 'MEG1412', 'MEG1222', 'MEG1212', 'MEG1232', 'MEG0922', 
                            'MEG0932', 'MEG1242', 'MEG0912', 'MEG0942', 'MEG1022', 'MEG1032', 'MEG0812', 'MEG1012', 'MEG0112', 
                            'MEG0132', 'MEG0212', 'MEG0222', 'MEG0142', 'MEG1512', 'MEG0242', 'MEG0232', 'MEG1542', 'MEG1522', 
                            'MEG1612', 'MEG1622', 'MEG1532', 'MEG1422', 'MEG1312', 'MEG1322', 'MEG1442', 'MEG1432', 'MEG1342', 
                            'MEG1332', 'MEG2612', 'MEG2412', 'MEG2422', 'MEG2642', 'MEG2622', 'MEG2632', 'MEG0412', 'MEG0422', 
                            'MEG0632', 'MEG0442', 'MEG0432', 'MEG0712', 'MEG1812', 'MEG1822', 'MEG0742', 'MEG1632', 'MEG1842', 
                            'MEG1832', 'MEG2012', 'MEG1042', 'MEG1112', 'MEG1122', 'MEG0722', 'MEG1142', 'MEG1132', 'MEG0732', 
                            'MEG2212', 'MEG2222', 'MEG2242', 'MEG2232', 'MEG2442', 'MEG2022', 'MEG1642', 'MEG1722', 'MEG1712', 
                            'MEG1912', 'MEG1942', 'MEG1732', 'MEG2042', 'MEG1922', 'MEG1932', 'MEG1742', 'MEG2112', 'MEG2142', 
                            'MEG2432', 'MEG2522', 'MEG2532', 'MEG2312', 'MEG2322', 'MEG2512', 'MEG2032', 'MEG2342', 'MEG2332', 
                            'MEG2542', 'MEG2122', 'MEG2132', 'MEG0123', 'MEG0343', 'MEG0313', 'MEG0323', 'MEG0513', 'MEG0543', 
                            'MEG0333', 'MEG0523', 'MEG0533', 'MEG0613', 'MEG0643', 'MEG0623', 'MEG0823', 'MEG1413', 'MEG1223', 
                            'MEG1213', 'MEG1233', 'MEG0923', 'MEG0933', 'MEG1243', 'MEG0913', 'MEG0943', 'MEG1023', 'MEG1033', 
                            'MEG0813', 'MEG1013', 'MEG0113', 'MEG0133', 'MEG0213', 'MEG0223', 'MEG0143', 'MEG1513', 'MEG0243', 
                            'MEG0233', 'MEG1543', 'MEG1523', 'MEG1613', 'MEG1623', 'MEG1533', 'MEG1423', 'MEG1313', 'MEG1323', 
                            'MEG1443', 'MEG1433', 'MEG1343', 'MEG1333', 'MEG2613', 'MEG2413', 'MEG2423', 'MEG2643', 'MEG2623', 
                            'MEG2633', 'MEG0413', 'MEG0423', 'MEG0633', 'MEG0443', 'MEG0433', 'MEG0713', 'MEG1813', 'MEG1823', 
                            'MEG0743', 'MEG1633', 'MEG1843', 'MEG1833', 'MEG2013', 'MEG1043', 'MEG1113', 'MEG1123', 'MEG0723', 
                            'MEG1143', 'MEG1133', 'MEG0733', 'MEG2213', 'MEG2223', 'MEG2243', 'MEG2233', 'MEG2443', 'MEG2023', 
                            'MEG1643', 'MEG1723', 'MEG1713', 'MEG1913', 'MEG1943', 'MEG1733', 'MEG2043', 'MEG1923', 'MEG1933', 
                            'MEG1743', 'MEG2113', 'MEG2143', 'MEG2433', 'MEG2523', 'MEG2533', 'MEG2313', 'MEG2323', 'MEG2513', 
                            'MEG2033', 'MEG2343', 'MEG2333', 'MEG2543', 'MEG2123', 'MEG2133']
            }       
        }
        
        save_sensor_layouts(MEGIN_layouts, 'MEGIN')
    
    elif modality=='EEG' and device=='GES':
        GES_layouts = {
            "GES_EEG_All":{ 
                'EEG_All': ['FidNZ','FidT9', 'FidT10', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 
                            'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31', 'E32', 'E33', 
                            'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E48', 'E49', 'E50', 'E51', 
                            'E52', 'E53', 'E54', 'E55', 'E56', 'E57', 'E58', 'E59', 'E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69', 
                            'E70', 'E71', 'E72', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 
                            'E88', 'E89', 'E90', 'E91', 'E92', 'E93', 'E94', 'E95', 'E96', 'E97', 'E98', 'E99', 'E100', 'E101', 'E102', 'E103', 'E104', 
                            'E105', 'E106', 'E107', 'E108', 'E109', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120', 
                            'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128', 'Cz']
            }
        }
        save_sensor_layouts(GES_layouts, 'EEG_GES')


if __name__ == "__main__":
        
	create_layouts()