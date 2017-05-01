package learn_matrix;

import mikera.matrixx.Matrix;
import org.junit.Test;

/**
 * Created by jarvis on 2017. 4. 26..
 */
public class Learn_Matrix {

    // 행렬 곱셉. 그냥 눈으로 본다 ㅋ
    @Test
    public void learnVector() throws Exception{

        Matrix matrix = Matrix.create(2,2);
        matrix.setElements(1,2,3,4);

        System.out.println(matrix);

        Matrix matrix2 = Matrix.create(2,2);
        matrix2.setElements(5,6,7,8);
        System.out.println("---");
        System.out.println(matrix2);

        matrix.multiply(matrix2);

        System.out.println("---");
        System.out.println(matrix);
        System.out.println("---");
        System.out.println(matrix2);
    }

    @Test
    public void 행렬_잘못된_더하기() throws Exception{
        Matrix matrix = Matrix.create(2,2);
        matrix.add(1);
        matrix.add(2);
        System.out.println(matrix);
    }



    @Test
    public void 차원구하기() throws Exception{
        Matrix m = Matrix.create(1,2);
        m.setElements(5,6);
        System.out.println(m.getShape()[0]);
        System.out.println(m.getShape()[1]);

        System.out.println(m.getShape(0));
        System.out.println(m.getShape(1));
        System.out.println(new Matrix(2,5));
    }


    @Test
    public void 맥스위치구하기() throws Exception{

    }



}
