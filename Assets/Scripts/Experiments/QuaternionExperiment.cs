
using UnityEngine;

/* Todo:
 * Find better ways for the neural nets to interact with the joints
 * 
 * Calculate difference between target rotation and current rotation
 * Look at cases where they can converge, where they diverge
 * 
 * I could use quaternions at the extremes, at the limits, to measure distance to.
 */

public class QuaternionExperiment : MonoBehaviour {
    [SerializeField] private ConfigurableJoint _joint;

    private Transform _transform;
    private Transform _parent;

    private Vector3 _basePitchAxis;
    private Vector3 _basePitchVector;

    private void Awake() {
        RunTests();

        _transform = _joint.gameObject.GetComponent<Transform>();
        _parent = _joint.connectedBody != null ? _joint.connectedBody.gameObject.transform : null;

        _basePitchAxis = ParentInverseRotation() * _transform.right;
        _basePitchVector = ParentInverseRotation() * _transform.forward;
    }

    private void Update() {
        // With the below, x and y are relative to parent, z is relative to local
        //
        //        Quaternion targetRotation = Quaternion.identity;
        //        targetRotation *= Quaternion.Euler(
        //            Input.GetAxis("Vertical") * 90f,
        //            Input.GetAxis("Depth") * 90f,
        //            Input.GetAxis("Horizontal") * 90f);

        Quaternion targetRotation = Quaternion.identity;
        targetRotation *= Quaternion.Euler(0f, 0f, Input.GetAxis("Horizontal") * _joint.angularZLimit.limit);
        targetRotation *= Quaternion.Euler(0f, Input.GetAxis("Depth") * _joint.angularYLimit.limit, 0f);
        targetRotation *= Quaternion.Euler(-Input.GetAxis("Vertical") * Mathf.Max(Mathf.Abs(_joint.lowAngularXLimit.limit), _joint.lowAngularXLimit.limit), 0f, 0f);
        _joint.targetRotation = targetRotation;

        Vector3 curPitchVector = ParentInverseRotation() * _transform.forward;
        float pitchAngle = Vector3.SignedAngle(
            curPitchVector,
            _basePitchVector,
            _basePitchAxis);

        Debug.DrawRay(Vector3.right, curPitchVector, Color.cyan);
        Debug.DrawRay(Vector3.right, _basePitchVector, Color.blue);
        Debug.DrawRay(Vector3.right, _basePitchAxis, Color.red);

        Debug.Log(pitchAngle);


        Quaternion positiveZLimit = Quaternion.AngleAxis(_joint.angularZLimit.limit, Vector3.forward);
        float angleToLimit = Quaternion.Angle(_joint.transform.rotation, positiveZLimit);
    }

    private Quaternion ParentInverseRotation() {
        Quaternion parentInverse = _parent != null ? Quaternion.Inverse(_parent.rotation) : Quaternion.identity;
        return parentInverse;
    }




    private static void RunTests() {
        Debug.Log(new Quaternion(0, 1, 2, 3) * new Quaternion(0, 1, 2, 3));
        Debug.Log(new MyQuaternion(0, 1, 2, 3) * new MyQuaternion(0, 1, 2, 3));

        Debug.Log(Quaternion.AngleAxis(90f, Vector3.forward));
        Debug.Log(MyQuaternion.AngleAxis(90f, Vector3.forward));

        Debug.Log(Quaternion.AngleAxis(75f, Vector3.forward + Vector3.up));
        Debug.Log(MyQuaternion.AngleAxis(75f, Vector3.forward + Vector3.up));

        Debug.Log(Quaternion.Euler(new Vector3(0f, 0f, 45f)));
        Debug.Log(MyQuaternion.Euler(new Vector3(0f, 0f, 45f)));
    }
}

public struct MyQuaternion {
    public float X, Y, Z, W;

    public MyQuaternion(float x, float y, float z, float w) {
        X = x;
        Y = y;
        Z = z;
        W = w;
    }

    public static MyQuaternion AngleAxis(float angle, Vector3 axis) {
        float angleRadHalf = Mathf.Deg2Rad * angle / 2.0f;
        float sinAngle = Mathf.Sin(angleRadHalf);
        axis.Normalize();
        return new MyQuaternion(
            Mathf.Cos(angleRadHalf),
            sinAngle * axis.z,
            sinAngle * axis.y,
            sinAngle * axis.x
        );
    }

    public static MyQuaternion Euler(Vector3 angles) {
        return AngleAxis(angles.x, Vector3.right) * AngleAxis(angles.z, Vector3.forward);
    }

    //    public static MyQuaternion Euler(float pitch, float roll, float yaw) {
    //        MyQuaternion q;
    //        // Abbreviations for the various angular functions
    //        float cy = Mathf.Cos(yaw * 0.5f);
    //        float sy = Mathf.Sin(yaw * 0.5f);
    //        float cr = Mathf.Cos(roll * 0.5f);
    //        float sr = Mathf.Sin(roll * 0.5f);
    //        float cp = Mathf.Cos(pitch * 0.5f);
    //        float sp = Mathf.Sin(pitch * 0.5f);
    //
    //        q.W = cy * cr * cp + sy * sr * sp;
    //        q.X = cy * sr * cp - sy * cr * sp;
    //        q.Y = cy * cr * sp + sy * sr * cp;
    //        q.Z = sy * cr * cp - cy * sr * sp;
    //        return q;
    //    }

    public static MyQuaternion Multiply(MyQuaternion a, MyQuaternion b) {
        return new MyQuaternion(
            a.W * b.W - a.Z * b.Z - a.Y * b.Y - a.X * b.X,
            a.W * b.Z + a.Z * b.W + a.Y * b.X - a.X * b.Y,
            a.W * b.Y - a.Z * b.X + a.Y * b.W + a.X * b.Z,
            a.W * b.X + a.Z * b.Y - a.Y * b.Z + a.X * b.W
        );
    }

    public static MyQuaternion operator * (MyQuaternion a, MyQuaternion b) {
        return new MyQuaternion(
            a.W * b.W - a.Z * b.Z - a.Y * b.Y - a.X * b.X,
            a.W * b.Z + a.Z * b.W + a.Y * b.X - a.X * b.Y,
            a.W * b.Y - a.Z * b.X + a.Y * b.W + a.X * b.Z,
            a.W * b.X + a.Z * b.Y - a.Y * b.Z + a.X * b.W
        );
    }

    public override string ToString() {
        return string.Format("({0:F1}, {1:F1}, {2:F1}, {3:F1})", W, Z, Y, X);
    }

    public static readonly MyQuaternion Identity = new MyQuaternion(0f, 0f, 0f, 1f);
}

public static class QuaternionHelper {
    /* Unity's implementation, straight from the source, without the double/float precision casts for
     * to make it readable. It works exactly the same as my code above, except the members are
     * reversed in meaning: W is the real/scalar part instead of X.
     */
    public static Quaternion Multiply(Quaternion lhs, Quaternion rhs) {
        return new Quaternion(
            (lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y),
            (lhs.w * rhs.y + lhs.y * rhs.w + lhs.z * rhs.x - lhs.x * rhs.z),
            (lhs.w * rhs.z + lhs.z * rhs.w + lhs.x * rhs.y - lhs.y * rhs.x),
            (lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z));
    }
}
