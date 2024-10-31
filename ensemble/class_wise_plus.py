import pandas as pd
import os
from mAP50_calculator import calculate_map50

def merge_bus_warning_values(csv1_path, csv2_path, output_path):
    # Load the first CSV file
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Define bus_warning class ID (index can vary based on class definition)
    BUS_WARNING_CLASS_ID = 13

    # Merge bus_warning values from the second CSV to the first CSV
    for idx, row in df1.iterrows():
        prediction_string1 = row['PredictionString']
        prediction_string2_list = df2[df2['image_id'] == row['image_id']]['PredictionString'].tolist()

        if not prediction_string2_list or pd.isna(prediction_string2_list[0]):
            continue

        prediction_string2 = prediction_string2_list[0]
        predictions1 = str(prediction_string1).split()
        predictions2 = str(prediction_string2).split()

        # Filter only bus_warning class predictions from the second CSV
        if len(predictions2) > 1:
            predictions2 = [predictions2[i:i + 6] for i in range(0, len(predictions2), 6)]
            bus_warning_predictions = [pred for pred in predictions2 if int(pred[0]) == BUS_WARNING_CLASS_ID]
        else:
            bus_warning_predictions = []

        # Parse predictions from the first CSV
        if len(predictions1) > 1:
            predictions1 = [predictions1[i:i + 6] for i in range(0, len(predictions1), 6)]
        else:
            predictions1 = []

        # Create final PredictionString by keeping original predictions and adding bus_warning predictions
        final_predictions = predictions1 + bus_warning_predictions
        df1.at[idx, 'PredictionString'] = ' '.join([' '.join(pred) for pred in final_predictions])

    # Save the result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df1.to_csv(output_path, index=False)
    print(f"Merged result saved to {output_path}")

if __name__ == "__main__":
    csv1_path = '/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/ensemble/output/csv/Co-DETR(Obj365, 3ep)_val.csv'
    csv2_path = '/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/ensemble/output/csv/co_dino_swin_l_o365_custom_2048_oversampling_val.csv'
    output_path = '/data/ephemeral/home/jiwan/2024-Autonomous-Driving-Artificial-Intelligence-Challenge/ensemble/output/csv/merged_bus_warning_plus.csv'

    # Merge bus_warning values from the second CSV to the first CSV
    merge_bus_warning_values(csv1_path, csv2_path, output_path)
