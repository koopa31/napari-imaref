package kovac.maths;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import kovac.res.util.MathUtils;

/**
 * Classe originale pour l'ajustement d'un ellipsoïde.
 */
public class EllipsoidAlgorithm {

    private static double gamma = 10.0;
    private static int nbIterations = 10000;

    private List<double[]> basePoints;
    private Matrix basePointsMatrix;

    /**
     * Constructeur prenant en entrée une liste de points en trois dimensions.
     */
    public EllipsoidAlgorithm(List<double[]> basePoints) {
        this.basePoints = basePoints;
        int numPoints = basePoints.size();
        this.basePointsMatrix = new Matrix(3, numPoints);

        for (int i = 0; i < numPoints; i++) {
            double[] point = basePoints.get(i);
            for (int j = 0; j < 3; j++) {
                this.basePointsMatrix.set(j, i, point[j]);
            }
        }
    }

    /**
     * Méthode principale de calcul de l'ellipse à partir des points donnés.
     */
    public double[] calculate() {
        // Debug : affichage des points de base
        System.out.println("Points de base :");
        for (double[] point : basePoints) {
            System.out.println(Arrays.toString(point));
        }

        Matrix centered = centerPoints(basePointsMatrix);
        // Debug : affichage des points centrés
        System.out.println("Points centrés :");
        centered.print(10, 6);

        Matrix K = computeK(centered);
        // Debug : affichage de la matrice K
        System.out.println("Matrice K :");
        K.print(10, 6);

        Matrix M = getM(K);
        // Debug : affichage de la matrice M
        System.out.println("Matrice M :");
        M.print(10, 6);

        Matrix q0 = initializeQ0(centered);
        // Debug : affichage de q0 initial
        System.out.println("q0 initial :");
        q0.print(10, 6);

        return runDouglasRachford(M, q0);
    }

    /**
     * Centre les points pour qu'ils soient exprimés par rapport au centre de masse.
     */
    private Matrix centerPoints(Matrix points) {
        double[] centerOfMass = new double[3];
        for (int i = 0; i < 3; i++) {
            centerOfMass[i] = MathUtils.mean(points.getArray()[i]);
        }

        // Debug : affichage du centre de masse
        System.out.println("Centre de masse : " + Arrays.toString(centerOfMass));

        Matrix centered = new Matrix(3, points.getColumnDimension());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < points.getColumnDimension(); j++) {
                centered.set(i, j, points.get(i, j) - centerOfMass[i]);
            }
        }
        return centered;
    }

    /**
     * Calcule la matrice K nécessaire pour l'algorithme Douglas-Rachford.
     */
    private Matrix computeK(Matrix centeredPoints) {
        int numPoints = centeredPoints.getColumnDimension();
        Matrix K = new Matrix(10, 10);

        for (int i = 0; i < numPoints; i++) {
            double[] p = { centeredPoints.get(0, i), centeredPoints.get(1, i), centeredPoints.get(2, i) };
            double[] q = {
                    p[0] * p[0], p[1] * p[1], p[2] * p[2],
                    Math.sqrt(2) * p[0] * p[1], Math.sqrt(2) * p[0] * p[2], Math.sqrt(2) * p[1] * p[2],
                    Math.sqrt(2) * p[0], Math.sqrt(2) * p[1], Math.sqrt(2) * p[2], 1.0
            };

            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < 10; k++) {
                    K.set(j, k, K.get(j, k) + q[j] * q[k]);
                }
            }
        }
        return K;
    }

    /**
     * Initialisation de la solution initiale q0.
     */
    private Matrix initializeQ0(Matrix points) {
        Matrix q0 = new Matrix(10, 1);
        q0.set(0, 0, 1.0 / 3.0);
        q0.set(1, 0, 1.0 / 3.0);
        q0.set(2, 0, 1.0 / 3.0);
        q0.set(9, 0, -1.0);
        return q0;
    }

    /**
     * Algorithme de Douglas-Rachford pour trouver la solution q.
     */
    private double[] runDouglasRachford(Matrix M, Matrix q0) {
        Matrix p = q0.copy();
        Matrix q = q0.copy();

        for (int iter = 0; iter < nbIterations; iter++) {
            q = proxf2(p);
            Matrix update = proxf1(M, q.times(2).minus(p)).minus(q);
            p = p.plus(update);

            // Debug : affichage des valeurs à chaque itération
            System.out.println("Itération " + iter + " :");
            System.out.println("q = " + Arrays.toString(q.getColumnPackedCopy()));

            if (update.normF() < 1e-8 * p.normF()) {
                System.out.println("Convergence atteinte à l'itération " + iter);
                break;
            }
        }
        return q.getColumnPackedCopy();
    }

    /**
     * Fonction de projection sur l'ensemble des solutions acceptables.
     */
    private Matrix proxf2(Matrix p) {
        double[][] A = new double[3][3];
        A[0][0] = p.get(0, 0);
        A[1][1] = p.get(1, 0);
        A[2][2] = p.get(2, 0);
        A[0][1] = A[1][0] = p.get(3, 0) / Math.sqrt(2);
        A[0][2] = A[2][0] = p.get(4, 0) / Math.sqrt(2);
        A[1][2] = A[2][1] = p.get(5, 0) / Math.sqrt(2);

        EigenvalueDecomposition eig = new EigenvalueDecomposition(new Matrix(A));
        Matrix V = eig.getV();
        double[] D = eig.getRealEigenvalues();
        double sum = 0;

        for (double d : D) {
            sum += Math.max(d, 0);
        }

        Matrix diagS = new Matrix(3, 3);
        for (int i = 0; i < 3; i++) {
            diagS.set(i, i, D[i] / sum);
        }

        Matrix Q = V.times(diagS).times(V.transpose());
        double[] newP = {
                Q.get(0, 0), Q.get(1, 1), Q.get(2, 2),
                Math.sqrt(2) * Q.get(0, 1), Math.sqrt(2) * Q.get(0, 2), Math.sqrt(2) * Q.get(1, 2),
                p.get(6, 0), p.get(7, 0), p.get(8, 0), p.get(9, 0)
        };

        return new Matrix(newP, 10);
    }

    /**
     * Fonction de projection sur l'ensemble des matrices définies positives.
     */
    private Matrix proxf1(Matrix M, Matrix q) {
        return M.solve(q);
    }

    /**
     * Calcule la matrice M à partir de K.
     */
    private Matrix getM(Matrix K) {
        Matrix I = Matrix.identity(K.getRowDimension(), K.getColumnDimension());
        return K.times(gamma).plus(I);
    }
}
