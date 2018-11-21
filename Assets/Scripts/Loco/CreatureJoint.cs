using UnityEngine;
using Ramjet;

/* 
 * Derivate useful set of basis quaterions from the joint limits
 * 
 * First, use them to get good measures of joint state. Then, use
 * them to express the action space of the net too. Slerp.
 * 
 * By doing Quaternion.Angle to a basis quaternion, twist interferes.
 * By comparing vector dots instead measurements with respect to your
 * basis become twist invariant.
 * 
 * Todo: twist
 * Todo: Account for x low/high asymmetry
 */

public class CreatureJoint : MonoBehaviour, ISensor, IActuator {
    private ConfigurableJoint _joint;
    private Transform _transform;
    private Transform _parent;
    private float[] _sensorActivations;
    private float[] _actuatorActivations;

    private Quaternion _base;
    private Quaternion _minX;
    private Quaternion _maxX;
    private Quaternion _maxY;

    private void Awake() {
        _joint = gameObject.GetComponent<ConfigurableJoint>();
        _transform = gameObject.GetComponent<Transform>();
        _parent = _joint.connectedBody != null ? _joint.connectedBody.gameObject.transform : null;

        _base = ParentInverseRotation() * _transform.rotation;

        _minX = Quaternion.Euler(_joint.lowAngularXLimit.limit, 0f, 0f);
        _maxX = Quaternion.Euler(_joint.highAngularXLimit.limit, 0f, 0f);
        _maxY = Quaternion.Euler(0f, _joint.angularYLimit.limit, 0f);

        _sensorActivations = new float[GetNumFreedoms() * 2];
        _actuatorActivations = new float[GetNumFreedoms()];
    }

    private int GetNumFreedoms() {
        int num = 0;
        num += _joint.angularXMotion != ConfigurableJointMotion.Locked ? 1 : 0;
        num += _joint.angularYMotion != ConfigurableJointMotion.Locked ? 1 : 0;
        num += _joint.angularZMotion != ConfigurableJointMotion.Locked ? 1 : 0;
        return num;
    }

    public void OnReset() {
        Utils.Zero(_sensorActivations);
        Utils.Zero(_actuatorActivations);
        FixedUpdate();
    }

    private void FixedUpdate() {
        //        Quaternion localRotation = ParentInverseRotation() * _transform.rotation;
        //        _sensorActivations[0] = Quaternion.Angle(
        //            localRotation, _base) / 90f;

        /* Read joint state for consumption by neural networks */

        Vector3 worldAnchor = _transform.TransformPoint(_joint.anchor);
        Vector3 worldBase = _parent.rotation * _base * Vector3.forward;
        Vector3 worldXMin = _parent.rotation * _base * _minX * Vector3.forward;
        Vector3 worldXMax = _parent.rotation * _base * _maxX * Vector3.forward;
        Vector3 worldYMin = _parent.rotation * _base * Quaternion.Inverse(_maxY) * Vector3.forward;
        Vector3 worldYMax = _parent.rotation * _base * _maxY * Vector3.forward;
        
        // Todo: normalize the total energy of this set of angles

        float maxXAngle = Mathf.Abs(_joint.lowAngularXLimit.limit - _joint.highAngularXLimit.limit);
        float maxYAngle = _joint.angularYLimit.limit * 2f;
        float dotBase = 1f - Mathf.Clamp01(Vector3.Angle(_transform.forward, worldBase) / maxXAngle);
        float dotXMin = 1f - Mathf.Clamp01(Vector3.Angle(_transform.forward, worldXMin) / maxXAngle);
        float dotXMax = 1f - Mathf.Clamp01(Vector3.Angle(_transform.forward, worldXMax) / maxXAngle);
        float dotYMin = 1f - Mathf.Clamp01(Vector3.Angle(_transform.forward, worldYMin) / maxYAngle);
        float dotYMax = 1f - Mathf.Clamp01(Vector3.Angle(_transform.forward, worldYMax) / maxYAngle);

//        dotBase = dotBase * dotBase;
//        dotXMin = dotXMin * dotXMin;
//        dotXMax = dotXMax * dotXMax;
//        dotYMin = dotYMin * dotYMin;
//        dotYMax = dotYMax * dotYMax;


        Debug.DrawRay(worldAnchor, worldBase * (dotBase), new Color(1f, 1f, 1f, dotBase));
        Debug.DrawRay(worldAnchor, worldXMin * (dotXMin), new Color(1f, 0f, 0f, dotXMin));
        Debug.DrawRay(worldAnchor, worldXMax * (dotXMax), new Color(1f, 0f, 0f, dotXMax));
        Debug.DrawRay(worldAnchor, worldYMin * (dotYMin), new Color(0f, 1f, 1f, dotYMin));
        Debug.DrawRay(worldAnchor, worldYMax * (dotYMax), new Color(0f, 1f, 1f, dotYMax));

        int i = 0;
        
        if (_joint.angularZMotion != ConfigurableJointMotion.Locked) {
            // Todo: twist
            _sensorActivations[i * 2 + 0] = dotBase;
            _sensorActivations[i * 2 + 1] = 0f;
            i++;
        }
        if (_joint.angularYMotion != ConfigurableJointMotion.Locked) {
            _sensorActivations[i * 2 + 0] = dotYMin;
            _sensorActivations[i * 2 + 1] = dotYMax;
            i++;
        }
        if (_joint.angularXMotion != ConfigurableJointMotion.Locked) {
            _sensorActivations[i * 2 + 0] = dotXMin;
            _sensorActivations[i * 2 + 1] = dotXMax;
            i++;
        }

        /* Convert network output to joint target rotation */

        // Todo: Order of rotations, parameterization of asymmetric x axis
        Quaternion targetRotation = Quaternion.identity;
        i = 0;
        if (_joint.angularZMotion != ConfigurableJointMotion.Locked) {
//            _actuatorActivations[i] = 0f;
            targetRotation =
                targetRotation *
                Quaternion.Euler(0f, 0f, _actuatorActivations[i] * _joint.angularZLimit.limit);
                //* targetRotation;
            i++;
        }
        if (_joint.angularYMotion != ConfigurableJointMotion.Locked) {
//            _actuatorActivations[i] = Mathf.Sin(Time.time * Mathf.PI * 0.5f);
            targetRotation =
                targetRotation *
                Quaternion.Euler(0f, _actuatorActivations[i] * _joint.angularYLimit.limit, 0f);
                //* targetRotation;
            i++;
        }
        if (_joint.angularXMotion != ConfigurableJointMotion.Locked) {
//            _actuatorActivations[i] = Mathf.Sin(Time.time * Mathf.PI * 0.25f);
            targetRotation =
                targetRotation *
                Quaternion.Euler(
                    _actuatorActivations[i] *
                    Mathf.Max(Mathf.Abs(_joint.lowAngularXLimit.limit), _joint.highAngularXLimit.limit), 0f, 0f);
            //* targetRotation;
        }
        _joint.targetRotation = targetRotation;
    }

    private Quaternion ParentInverseRotation() {
        Quaternion parentInverse = _parent != null ? Quaternion.Inverse(_parent.rotation) : Quaternion.identity;
        return parentInverse;
    }

    int ISensor.SensorCount {
        get { return _sensorActivations.Length; }
    }

    public float Get(int index) {
        return _sensorActivations[index];
    }

    int IActuator.ActuatorCount {
        get { return _actuatorActivations.Length; }
    }

    public void Set(int index, float value) {
        _actuatorActivations[index] = value;
    }
}
 