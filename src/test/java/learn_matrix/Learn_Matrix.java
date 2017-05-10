package learn_matrix;

import com.arahansa.data.Grad;
import com.arahansa.neuralnet.J06_TwoLayerNet;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import mikera.vectorz.Vector;
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
    public void 행렬_더하기() throws Exception{
        Matrix matrix = Matrix.create(2,2);
        matrix.add(1);
        matrix.add(2);
        System.out.println(matrix);
    }



    @Test
    public void 차원구하기() throws Exception{
        Matrix m = Matrix.create(1,2);
        m.setElements(5,6);

        System.out.println(m.getShape(0));
        System.out.println(m.getShape(1));

        System.out.println(m);
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



    @Test
    public void matrixNewTest() throws Exception{

        Matrix matrix = Matrix.create(2,2);
        matrix.setElements(1,2,3,4);

        Matrix u = addOneRow(matrix);


        log.info("u : {}", u);
        u.setRow(2, Vector.of(5,6));
        log.info("u : {}", u);
    }

    private Matrix addOneRow(Matrix x){
        Matrix newM = Matrix.create(x.getShape(0)+1, x.getShape(1));

        for(int i=0;i<x.getShape(0);i++){
            newM.setRow(i,x.getRow(i));
        }
        return newM;
    }






}
