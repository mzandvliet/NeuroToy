using UnityEngine;

// Senses angle to goal
public class AngleSensor : MonoBehaviour, ISensor {
    private Vector3 _goal;

    private Transform _transform;
    private float[] _activations;

    public Vector3 Goal {
        get { return _goal; }
        set { _goal = value; }
    }

    private void Awake() {
        _transform = gameObject.GetComponent<Transform>();
        _activations = new float[2];
    }

    public void OnReset() {
        Utils.Zero(_activations);
    }

    private void FixedUpdate() {
//        Vector3 goalDiff = Vector3.ClampMagnitude(_goal - _transform.position, 10f);
//        goalDiff /= 10f;
//
//        _activations[0] = goalDiff.x;
//        _activations[1] = goalDiff.y;
//        _activations[2] = goalDiff.z;

        _activations[0] = Vector3.Dot(_transform.forward, _goal);
        _activations[1] = Vector3.Dot(_transform.right, _goal);
    }

    int ISensor.SensorCount {
        get { return _activations.Length; }
    }

    public float Get(int index) {
        return _activations[index];
    }
}