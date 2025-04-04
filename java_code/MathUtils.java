package kovac.res.util;

public class MathUtils {

    public static double[] getCenterOfMass(double[][] points) {
        double[] centerOfMass = new double[3];
        for (int i = 0; i < 3; i++) {
            double sum = 0.0;
            for (double value : points[i]) {
                sum += value;
            }
            centerOfMass[i] = sum / points[i].length;
        }
        return centerOfMass;
    }
}
