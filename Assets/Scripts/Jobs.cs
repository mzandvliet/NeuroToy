using Unity.Jobs;
using Unity.Collections;
using UnityEngine;

public struct AddJob : IJob {
    public NativeArray<float> A;
    public NativeArray<float> B;
    public NativeArray<float> R;

    public void Execute() {
        if (A.Length != B.Length || A.Length != R.Length) {
            Debug.LogError("Arrays need to be of same length.");
            return;
        }

        for (int i = 0; i < A.Length; i++) {
            R[i] = A[i] + B[i];
        }
    }
}

public struct SubtractJob : IJob {
    public NativeArray<float> A;
    public NativeArray<float> B;
    public NativeArray<float> R;

    public void Execute() {
        if (A.Length != B.Length || A.Length != R.Length) {
            Debug.LogError("Arrays need to be of same length.");
            return;
        }

        for (int i = 0; i < A.Length; i++) {
            R[i] = A[i] - B[i];
        }
    }
}

public struct HadamardJob : IJob {
    public NativeArray<float> A;
    public NativeArray<float> B;
    public NativeArray<float> R;

    public void Execute() {
        if (A.Length != B.Length || A.Length != R.Length) {
            Debug.LogError("Arrays need to be of same length.");
            return;
        }

        for (int i = 0; i < A.Length; i++) {
            R[i] = A[i] * B[i];
        }
    }
}

public struct DotJob : IJob {
    public NativeArray<float> A;
    public NativeArray<float> B;
    public float R;

    public void Execute() {
        if (A.Length != B.Length) {
            Debug.LogError("Arrays need to be of same length.");
            return;
        }

        for (int i = 0; i < A.Length; i++) {
            R += A[i] * B[i];
        }

        R /= (float)A.Length;
    }
}