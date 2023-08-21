
import functools

#averages all of the given dicts
def average(dicts):
    result_dict = {}
    for key in dicts[0].keys():
        current_key_values = [dictionary[key] for dictionary in dicts]
        current_key_sum = functools.reduce( lambda accumulator, value: accumulator + value, current_key_values )
        current_key_average = current_key_sum / len(dicts)
        result_dict[key] = current_key_average
    return result_dict