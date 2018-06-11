using UnityEngine;

public class VelocitySensor : MonoBehaviour, ISensor {
    private Transform _transform;
    private Rigidbody _body;

    private float[] _activation;

    private Vector3 _lastLocalLinearVelocity;
    private Vector3 _lastLocalAngularVelocity;

    private void Awake() {
        _transform = gameObject.GetComponent<Transform>();
        _body = gameObject.GetComponent<Rigidbody>();
        _activation = new float[1];
    }

    public void OnReset() {
        _lastLocalLinearVelocity = Vector3.zero;
        _lastLocalAngularVelocity = Vector3.zero;
    }

    private void FixedUpdate() {
        // Todo: Interpolation rates are framerate dependent now
        const float lerpSpeed = 0.03f;

        Vector3 localLinearVelocity = _transform.InverseTransformDirection(_body.velocity);
        localLinearVelocity = Vector3.ClampMagnitude(localLinearVelocity * 0.25f, 1f);
        localLinearVelocity = Vector3.Lerp(_lastLocalLinearVelocity, localLinearVelocity, lerpSpeed);
        _lastLocalLinearVelocity = localLinearVelocity;

        Vector3 localAngularVelocity = _transform.InverseTransformDirection(_body.angularVelocity);
        localAngularVelocity = Vector3.ClampMagnitude(localAngularVelocity, 1f);
        localAngularVelocity = Vector3.Lerp(_lastLocalAngularVelocity, localAngularVelocity, lerpSpeed);
        _lastLocalAngularVelocity = localAngularVelocity;

        Debug.DrawRay(_transform.position, _transform.TransformDirection(_lastLocalLinearVelocity) * 10f, Color.blue);
        
        _activation[0] = localAngularVelocity.y;
    }

    int ISensor.SensorCount {
        get { return _activation.Length; }
    }

    public float Get(int index) {
        return _activation[index];
    }
}