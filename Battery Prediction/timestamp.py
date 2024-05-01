import os
import numpy as np
import pandas as pd

def average50() :
    # Load the data
    data = pd.read_csv('magic_file.csv')

    # Process the data (e.g., sorting)
    file_data = data.sort_values(by='timestamp').to_numpy()
    print(f"The number of NaN elements before cleaning is : {np.sum(np.isnan(file_data))}")
    file_data = np.nan_to_num(file_data, nan=0)
    rows, cols = file_data.shape # (rows, cols)
    print(f"The number of NaN elements after cleaning is : {np.sum(np.isnan(file_data))}")
    print(f"The data sheet has {rows} rows and {cols} cols")

    nrows = rows // 50 + 1
    processed_array = np.zeros((nrows, cols))

    for i in range(0,rows, 50):
        if i + 50 > rows:
            sub_data = file_data[rows - 50:rows]
        else:
            sub_data = file_data[i:i+50]
        new_row = np.sum(sub_data, axis=0) / np.count_nonzero(sub_data, axis=0)
        processed_array[i // 50] += new_row

    print(processed_array[-5:, -5])
    print(f"The number of nan elements after preprocessing is : {np.sum(np.isnan(processed_array))}")
    new_df = pd.DataFrame(data=processed_array, columns=data.columns)

    # Save the new DataFrame to a CSV file
    new_df.to_csv('output.csv', index=False)

def moving_average():
    data = pd.read_csv('output.csv.csv')
    file_data = data.sort_values(by='timestamp').to_numpy()
    print(f"The number of NaN elements before cleaning is : {np.sum(np.isnan(file_data))}")
    file_data = np.nan_to_num(file_data, nan=0)
    rows, cols = file_data.shape  # (rows, cols)
    processed_array = np.zeros((rows, cols))
    print(f"The number of NaN elements after cleaning is : {np.sum(np.isnan(file_data))}")
    print(f"The data sheet has {rows} rows and {cols} cols")

    for i in range(0, rows):
        if i < 2:
            new_row = file_data[i:i+5]
        elif i > rows - 3 :
            new_row = file_data[rows-5:]
        else:
            new_row = file_data[i - 2:i+3]
        new_row = np.sum(new_row, axis=0) / np.count_nonzero(new_row, axis = 0)
        for j in range(0, cols):
            if file_data[i,j] == 0 :
                processed_array[i, j] = new_row[j]
            else:
                processed_array[i,j] = file_data[i,j]

    new_df = pd.DataFrame(data=processed_array, columns=data.columns)

    # Convert 'remaining' to percentage
    if 'remaining' in new_df.columns:
        new_df['remaining'] = new_df['remaining'] / max(new_df['remaining']) * 100

    # Save the new DataFrame to a CSV file
    new_df.to_csv('average.csv', index=False)

if __name__ == "__main__" :
    average50()
    moving_average()