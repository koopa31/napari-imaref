package kovac.res.quadric;

public class QuadricExpression {

    private final double[] coefficients;

    public QuadricExpression(double[] coefficients) {
        if (coefficients.length != 10) {
            throw new IllegalArgumentException("Requires 10 coefficients");
        }
        this.coefficients = coefficients.clone();
    }

    public double[][] getMatrixA() {
        return new double[][]{
            {coefficients[0], coefficients[3] / Math.sqrt(2), coefficients[4] / Math.sqrt(2)},
            {coefficients[3] / Math.sqrt(2), coefficients[1], coefficients[5] / Math.sqrt(2)},
            {coefficients[4] / Math.sqrt(2), coefficients[5] / Math.sqrt(2), coefficients[2]}
        };
    }

    public double[] getVectorB() {
        return new double[]{coefficients[6] / Math.sqrt(2), coefficients[7] / Math.sqrt(2), coefficients[8] / Math.sqrt(2)};
    }

    public double getScalarC() {
        return coefficients[9];
    }

    public double[] getCoefficients() {
        return coefficients.clone();
    }
}
