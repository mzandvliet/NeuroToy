using UnityEngine;
using Old;

// Todo: Grabbing requires energy, depletes. Net can read back energy.
public class CreatureClaw : MonoBehaviour, ISensor, IActuator {
    private ConfigurableJoint _joint;
    private Transform _transform;

    private float[] _sensorActivations;
    private float[] _actuatorActivations;

    private float _radius = 0.1f;

    public ConfigurableJoint Joint {
        get { return _joint; }
    }

    private int _layerMask;

    private void Awake() {
        _transform = gameObject.GetComponent<Transform>();

        _joint = gameObject.AddComponent<ConfigurableJoint>();
        Release();
        _joint.angularXMotion = ConfigurableJointMotion.Free;
        _joint.angularYMotion = ConfigurableJointMotion.Free;
        _joint.angularYMotion = ConfigurableJointMotion.Free;
        _joint.linearLimit = new SoftJointLimit() {
            limit = 0.05f,
        };
        _joint.linearLimitSpring = new SoftJointLimitSpring() {
            spring = 100f,
            damper = 20f
        };

        _sensorActivations = new float[1];
        _actuatorActivations = new float[1];

        _layerMask = ~LayerMask.GetMask("Creature");
    }

  

    public void OnReset() {
        Release();
        Utils.Set(_sensorActivations, 0f);
        Utils.Set(_actuatorActivations, 0f);
        FixedUpdate();
    }

    private void FixedUpdate() {
        bool isLatched = IsLatched();

        if (!isLatched) {
            if (_actuatorActivations[0] > 0.1f) {
                TryLatch();
            }
        }
        else {
            if (_actuatorActivations[0] < -0.1f) {
                Release();
            }
        }

        _sensorActivations[0] = Mathf.Lerp(_sensorActivations[0], isLatched ? 1f : 0f, Time.fixedDeltaTime * 10f);
    }

    private bool TryLatch() {
        if (Physics.CheckSphere(_transform.TransformPoint(_joint.anchor), _radius, _layerMask)) {
            Latch();
            return true;
        }
        return false;
    }

    private bool IsLatched() {
        return _joint.xMotion == ConfigurableJointMotion.Limited;
    }

    private void Latch() {
        _joint.connectedAnchor = _transform.TransformPoint(_joint.anchor);
        _joint.xMotion = ConfigurableJointMotion.Limited;
        _joint.yMotion = ConfigurableJointMotion.Limited;
        _joint.zMotion = ConfigurableJointMotion.Limited;
    }

    private void Release() {
        _joint.xMotion = ConfigurableJointMotion.Free;
        _joint.yMotion = ConfigurableJointMotion.Free;
        _joint.zMotion = ConfigurableJointMotion.Free;
    }

    int ISensor.SensorCount {
        get {
            return _sensorActivations.Length;
        }
    }

    public float Get(int index) {
        return _sensorActivations[index];
    }

    int IActuator.ActuatorCount {
        get { return _actuatorActivations.Length; }
    }

    public void Set(int index, float value) {
        _actuatorActivations[index] = Mathf.Lerp(_actuatorActivations[index], value, Time.fixedDeltaTime * 5f);
    }

    private void OnDrawGizmos() {
        Gizmos.color = IsLatched() ? new Color(0.5f, 1f, 1f, 0.9f) : new Color(0.5f, 1f, 1f, 0.33f);
        Gizmos.DrawSphere(_transform.TransformPoint(_joint.anchor), _radius);
    }
}