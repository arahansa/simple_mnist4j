package learn_matrix;

import com.arahansa.data.Grad;
import com.arahansa.neuralnet.J06_TwoLayerNet;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.junit.Test;

/**
 * Created by jarvis on 2017. 4. 26..
 */
@Slf4j
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
    public void transpose() throws Exception{
        Matrix m = Matrix.create(3,3);
        m.setElements(1,2,3,4,5,6,7,8,9);

        m.transposeInPlace();
        log.info("after transpose : {}", m);
    }

    @Test
    public void minus() throws Exception{
        Matrix m = Matrix.create(3,3);
        m.setElements(1,2,3,4,5,6,7,8,9);


        Matrix copy = m.copy();
        mminus(copy);
        log.info("m copy : {}", copy);

        log.info("m : {}", m);
    }

    private void mminus(Matrix m){
        m.multiply(-1);
    }

    @Test
    public void gradient() throws Exception{

        J06_TwoLayerNet twoLayerNet = new J06_TwoLayerNet(3, 3, 3, null);

        Matrix m = Matrix.create(3,3);
        m.setElements(1,2,3,4,5,6,7,8,9);


        twoLayerNet.setW1(m);
        twoLayerNet.setW2(m);

        Grad gradient = twoLayerNet.gradient(m, m);
        log.info("gradient: {}", gradient);


    }




    /*'W2':
    ([[-3.99764437, -4.95268584, -5.04966979],
    [-3.99764437, -4.95268584, -5.04966979],
    [-3.99764437, -4.95268584, -5.04966979]]),
    'W1':
    ([[ -3.44383981e-13,  -1.92823723e-15,   0.00000000e+00],
     [ -6.88767962e-13,  -3.85647446e-15,   0.00000000e+00],
     [ -1.03315194e-12,  -5.78471169e-15,   0.00000000e+00]]),
    'b1':
    [ -3.44383981e-13,  -1.92823723e-15,   0.00000000e+00]), 'b2': array([-3.99764437, -4.95268584, -5.04966979])}*/






}
