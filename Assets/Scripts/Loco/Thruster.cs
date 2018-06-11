
using System;
using UnityEngine;

public class Thruster : MonoBehaviour, IActuator, ISensor {
    [SerializeField] private Rigidbody _rigidbody;
    [SerializeField] private float _maxForce;

    private Transform _transform;
    private float _actuatorActivation;

    public Rigidbody Rigidbody {
        get { return _rigidbody; }
        set { _rigidbody = value; }
    }

    public float MaxForce {
        get { return _maxForce; }
        set { _maxForce = value; }
    }

    public int ActuatorCount {
        get { return 1; }
    }

    public int SensorCount {
        get { return 1; }
    }

    public float Get(int index) {
        return _actuatorActivation; // So the net can know what its previous move was
    }

    public void OnReset() {
        _actuatorActivation = 0f;
    }

    public void Set(int index, float value) {
        _actuatorActivation = Mathf.Lerp(_actuatorActivation, value, Time.fixedDeltaTime * 10f);
    }

    private void Awake() {
        _transform = gameObject.GetComponent<Transform>();
    }

    private void FixedUpdate() {
        _rigidbody.AddForceAtPosition(_transform.forward * (_maxForce * _actuatorActivation), _transform.position, ForceMode.Force);
    }

    private void OnDrawGizmos() {
        Gizmos.color = Color.green;
        Gizmos.DrawRay(_transform.position, _transform.forward * -_actuatorActivation);
    }
}
