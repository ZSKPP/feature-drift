def parse_file(file_path):
    results = []
    with open(file_path, "r") as file:
        content = file.read().strip()

    blocks = content.split("\n\n")

    for block in blocks:
        lines = block.split("\n")
        file_name = lines[0].strip()
        metrics = []

        for line in lines[1:]:
            parts = line.split()
            metric_name = parts[0]
            values = list(map(int, parts[1:])) if len(parts) > 1 else []
            metrics.append([metric_name, values])

        results.append([file_name, metrics])

    return results


def merge_results(results1, results2):
    merged_results = []

    for block1, block2 in zip(results1, results2):
        file_name1, metrics1 = block1
        file_name2, metrics2 = block2

        if file_name1 != file_name2:
            raise ValueError(f"File names do not match: {file_name1} != {file_name2}")

        combined_metrics = metrics2 + metrics1
        merged_results.append([file_name1, combined_metrics])

    return merged_results


def split_detections(detectedIdx, trueIdx, epsilon):
    detectionTrue = []
    detectionFalse = []
    detectionMissed = []

    for detected in detectedIdx:
        if any(abs(detected - true) <= epsilon for true in trueIdx):
            detectionTrue.append(detected)
        else:
            detectionFalse.append(detected)

    for true in trueIdx:
        if not (any(abs(detected - true) <= epsilon for detected in detectedIdx)):
            detectionMissed.append(true)

    return detectionTrue, detectionFalse, detectionMissed


file_path1 = "D:\Tomek\Documents\Python\PROFESOR\Dryfty_10\FROUROUS_Recurring_Drifts.txt"   # Path to file with FROROUS results
file_path2 = "D:\Tomek\Documents\Python\PROFESOR\Dryfty_10\FBDD_Recurring_Drifts_10.txt"    # Path to file with FBDD results

realDriftIdx = [20000, 45000, 70000]    # [45000]  # [20000, 45000, 70000]                  # Real (reference) drift indices
epsilon = 2000                                                                              # Tolerance range for drift detection (absolute)

results1 = parse_file(file_path1)
results2 = parse_file(file_path2)

merged_results = merge_results(results1, results2)

fcount = len(merged_results)
dcount = len(merged_results[0][1])

tTP = [0] * dcount
tFP = [0] * dcount
tFN = [0] * dcount

for detectors in merged_results:
    #print(detectors[0])
    for det in range(0, len(detectors[1])):
        (
            detectedTrueDriftIdx,
            detectedFalseDriftIdx,
            detectedMissedDriftIdx,
        ) = split_detections(detectors[1][det][1], realDriftIdx, epsilon)
        TP = len(detectedTrueDriftIdx)
        FP = len(detectedFalseDriftIdx)
        FN = len(detectedMissedDriftIdx)
        #print(detectors[1][det][0], "\t", TP, "\t", FP, "\t", FN)
        tTP[det] = tTP[det] + TP
        tFP[det] = tFP[det] + FP
        tFN[det] = tFN[det] + FN

for det in range(0, dcount):
    print(merged_results[1][1][det][0], "\t", tTP[det]/fcount, "\t", tFP[det]/fcount, "\t", tFN[det]/fcount)
