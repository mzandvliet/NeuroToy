using UnityEngine;

[RequireComponent(typeof(Collider))]
public class CreatureCollider : MonoBehaviour, ISensor {
    private Collider _collider;
    private float _activation;
    private float _collisionImpulse;

    private MeshRenderer _renderer;

    private bool _collidedThisFrame;

    private void Awake() {
        _collider = gameObject.GetComponent<Collider>();
        _renderer = gameObject.GetComponentInChildren<MeshRenderer>();
    }

    public void OnReset() {
        _activation = 0f;
        _collisionImpulse = 0f;
    }

    public int SensorCount {
        get { return 1; }
    }

    public float Get(int index) {
        return _activation;
    }

    private void OnCollisionEnter(Collision collision) {
        _collisionImpulse = Mathf.Min(1f, collision.impulse.magnitude / 10f);
        _collidedThisFrame = true;
    }

    private void OnCollisionStay(Collision collision) {
        _collisionImpulse = Mathf.Min(1f, collision.impulse.magnitude / 10f);
        _collidedThisFrame = true;
    }

    private void FixedUpdate() {
        // Todo: Interpolation rates are framerate dependent now
        if (_collidedThisFrame) {
            _activation = Mathf.Lerp(_activation, _collisionImpulse, 0.05f);
        }
        else {
            _activation *= 0.95f;
        }

        _collidedThisFrame = false;
    }

    private void Update() {
        _renderer.material.color = new Color(1f, 1f-_activation, 1f-_activation, 1f);
    }
}