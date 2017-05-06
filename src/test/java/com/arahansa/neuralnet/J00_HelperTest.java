package com.arahansa.neuralnet;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import mikera.matrixx.impl.ColumnMatrix;
import mikera.matrixx.impl.RowMatrix;
import mikera.vectorz.AVector;
import mikera.vectorz.Vector;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.offset;
import static org.assertj.core.api.Java6Assertions.assertThat;

/**
 * Created by arahansa on 2017-05-01.
 */
@Slf4j
public class J00_HelperTest {

    @Test
    public void softmax_process() throws Exception{
        Matrix m = new Matrix(1, 3);
        m.setElements(0.3, 2.9, 4.0);

        log.info("m : {}", m);

        m.exp();
        log.info("m : {}", m);

        final double sum = m.elementSum();
        log.info("sum : {}", sum);

        m.divide(sum);

        log.info("after divide : {}", m);
    }

    @Test
    public void simple_softmax_overflow() throws Exception{
        Matrix m = new Matrix(1,3);
        m.setElements(1010, 1000, 990);
        log.info("max  : {} ", m.elementMax());

        m.exp();
        double sum = m.elementSum();
        m.divide(sum);
        log.info("after divide : {}", m); // Nan Nan Nan


        // 다시 최대값을 빼서 계산해본다.
        m.setElements(1010, 1000, 990);
        double max = m.elementMax();
        m.add(-max);
        log.info(" a - c : {}", m);

        m.exp();
        sum = m.elementSum();
        m.divide(sum);
        log.info("softmax a-c : {}", m);
    }

    @Test
    public void softmax() throws Exception{
        Matrix m = new Matrix(1,3);
        m.setElements(1010, 1000, 990);

        J00_Helper helper = new J00_Helper();
        final Matrix softmax = helper.softmax(m);
        log.info("softmax : {}", softmax);

        assertThat(0.99).isEqualTo(softmax.get(0,0), offset(0.02));
        assertThat(softmax.get(0,1)).isEqualTo(0.0000453, offset(0.02));
        assertThat(softmax.get(0,2)).isEqualTo(2.06E-9, offset(0.02));
    }


    @Test
    public void sigmoid_역수() throws Exception{
        Matrix m = new Matrix(1,4);
        m.setElements(1,2,3,4);
        m.reciprocal();
        log.info("reciprocal :{}", m);
    }

    @Test
    public void sigmoid() throws Exception{
        Matrix m = new Matrix(1,3);
        m.setElements(-1, 1, 2);

        J00_Helper helper = new J00_Helper();
        final Matrix sigmoid = helper.sigmoid(m);
        log.info("after sigmoid :{}", sigmoid);

        assertThat(sigmoid.get(0,0)).isEqualTo(0.26, offset(0.01));
        assertThat(sigmoid.get(0,1)).isEqualTo(0.73, offset(0.01));
        assertThat(sigmoid.get(0,2)).isEqualTo(0.88, offset(0.01));
    }


    // 가로줄 세로줄 곱하기 테스트
    @Test
    public void multiple2Matrix() throws Exception{
        RowMatrix rowMatrix = new RowMatrix(Vector.of(1,2,3,4,5));
        log.info("rowMatrix :{}", rowMatrix);

        final AVector row = rowMatrix.getRow(0);
        log.info("row : {}", row);

        ColumnMatrix columnMatrix = new ColumnMatrix(Vector.of(1,2,3,4,5));
        final AVector column = columnMatrix.getColumn(0);
        log.info("colum : {}", column);

        row.multiply(column);
        log.info("after multiple : {}", row);
        final double v = row.elementSum();
        log.info("sum : {}", v);
        assertThat(v).isEqualTo(55, offset(0.1));
    }

    // 두 행렬간의 곱 테스트
    @Test
    public void multipleTwoMatrix() throws Exception{
        Matrix m = Matrix.create(2,2);
        m.setElements(1,2,3,4);

        Matrix y = Matrix.create(2,3);
        y.setElements(1,2,3,4, 5,6);

        final Matrix multiplied = J00_Helper.multipleTwoMatrix(m, y);
        log.info("multiplied : {}", multiplied);

        Matrix answer = Matrix.create(2, 3);
        answer.setElements(9, 12, 15, 19, 26, 33);
        log.info("eq : {}", answer.equals(multiplied));
        assertThat(answer.equals(multiplied)).isTrue();
    }

    @Test
    public void 정규분포행렬얻기() throws Exception{
        log.info("정규분포 행렬 :{} ", J00_Helper.getRadNMatrix(2,2));
    }


    static class Counter{
        static int count;
        static void plus(){
            count++;
        }
    }
    @Test
    public void 랜덤중복되지않은숫자만들기() throws Exception{
        List<Integer> range = IntStream.rangeClosed(0, 99).boxed()
                .collect(Collectors.toCollection(ArrayList::new));
        Collections.shuffle(range);

        Counter counter = new Counter();
        range.subList(1, 10).forEach(n->{
            counter.plus();
            System.out.println(n);
        });
        System.out.println("after :"+counter.count);
    }

    @Test
    public void getBatchMatrixTest() throws Exception{
        Matrix x = Matrix.create(5, 10);
        Matrix t = Matrix.create(5, 5);


        x.setElements(IntStream.range(0,50).mapToDouble(Double::new).toArray());
        t.setElements(IntStream.range(0,25).mapToDouble(Double::new).toArray());

        log.info("x :{}", x);

        Map<String, Matrix> batchMatrix = J00_Helper.getBatchMatrix(x, t, 2);

        log.info("x_batch :{}", batchMatrix.get("x_batch"));
        log.info("t_batch :{}", batchMatrix.get("t_batch"));

    }


    @Test
    public void sum() throws Exception{
        Matrix x = Matrix.create(3,3);
        x.setElements(
                1,2,3,
                4,5,6,
                7,8,9
        );

        Matrix sum = J00_Helper.sum(x);
        log.info("sum : {}", sum);

        x = Matrix.create(3,1);

        x.setElements(1,4,7);
        sum = J00_Helper.sum(x);
        log.info("sum : {}", sum);

    }

    @Test
    public void sigmoid_gradTest() throws Exception{
        Matrix x = Matrix.create(3,3);
        x.setElements(
                1,2,3,
                4,5,6,
                7,8,9
        );
        /* 파이썬 정답
        [  1.96611933e-01   1.04993585e-01   4.51766597e-02]
        [  1.76627062e-02   6.64805667e-03   2.46650929e-03]
        [  9.10221180e-04   3.35237671e-04   1.23379350e-04]
        */
        Matrix sigmoid_grad = J00_Helper.sigmoid_grad(x);
        log.info("sigmoid grad: {}", sigmoid_grad);

        assertThat(sigmoid_grad.get(0,0)).isEqualTo(0.196, offset(0.001));
        assertThat(sigmoid_grad.get(0,1)).isEqualTo(0.104, offset(0.001));
        assertThat(sigmoid_grad.get(0,2)).isEqualTo(0.0451, offset(0.0001));

        assertThat(sigmoid_grad.get(1,0)).isEqualTo(0.0176, offset(0.001));
        assertThat(sigmoid_grad.get(1,1)).isEqualTo(0.0064, offset(0.001));
        assertThat(sigmoid_grad.get(1,2)).isEqualTo(0.0024, offset(0.0001));

        assertThat(sigmoid_grad.get(2,0)).isEqualTo(0.00091, offset(0.0001));
        assertThat(sigmoid_grad.get(2,1)).isEqualTo(0.00033, offset(0.0001));
        assertThat(sigmoid_grad.get(2,2)).isEqualTo(0.00012, offset(0.0001));



    }



    @Test
    public void divideTest() throws Exception{
        Matrix m = Matrix.create(3,3);
        m.setElements(8,8,8,8,8,8,8,8,8);

        Matrix c = Matrix.create(1,3);
        c.setElements(1,2,4);

        m.getRow(0).divide(c.getRow(0));
        assertThat(m.get(0,0)).isEqualTo(8.0, offset(0.01));
        assertThat(m.get(0,1)).isEqualTo(4.0, offset(0.01));
        assertThat(m.get(0,2)).isEqualTo(2.0, offset(0.01));
        log.info("m : {}", m);



        m.setElements(8,8,8,8,8,8,8,8,8);
        Matrix d = J00_Helper.divide(m, c);
        log.info("divide : {}", d);

        assertThat(d.get(0,0)).isEqualTo(8.0, offset(0.01));
        assertThat(d.get(0,1)).isEqualTo(4.0, offset(0.01));
        assertThat(d.get(0,2)).isEqualTo(2.0, offset(0.01));

        assertThat(d.get(1,0)).isEqualTo(8.0, offset(0.01));
        assertThat(d.get(1,1)).isEqualTo(4.0, offset(0.01));
        assertThat(d.get(1,2)).isEqualTo(2.0, offset(0.01));

        assertThat(d.get(2,0)).isEqualTo(8.0, offset(0.01));
        assertThat(d.get(2,1)).isEqualTo(4.0, offset(0.01));
        assertThat(d.get(2,2)).isEqualTo(2.0, offset(0.01));


    }


    @Test
    public void max0Dimen() throws Exception{
        Matrix m = Matrix.create(3,4);
        m.setElements(
                4,3,2,9,
                2,4,6,8,
                5,3,7,2
        );

        Matrix d = J00_Helper.max0Dimen(m);
        log.info("m : {}", d);

        // [5.0,4.0,7.0,9.0]
        assertThat(d.get(0, 0)).isEqualTo(5.0);
        assertThat(d.get(0, 1)).isEqualTo(4.0);
        assertThat(d.get(0, 2)).isEqualTo(7.0);
        assertThat(d.get(0, 3)).isEqualTo(9.0);
    }


    @Test
    public void transpose() throws Exception{
        Matrix x = Matrix.create(1,4);
        x.setElements(1,2,3,4);


        Matrix t = J00_Helper.transpose(x);
        log.info("t : {}", t);
        assertThat(t.isSameShape(Matrix.create(4,1))).isTrue();


        x = Matrix.create(3, 3);
        x.setElements(1,2,3,4,5,6,7,8,9);

        t = J00_Helper.transpose(x);
        assertThat(t.isSameShape(Matrix.create(3,3))).isTrue();
        log.info("t : {}", t);
        log.info("t shape : {}", t.getShape());

        x.setElements(1,4,7,2,5,8,3,6,9);
        assertThat(t.equals(x)).isTrue();

    }



    @Test
    public void argmaxTest() throws Exception{
        Matrix x = Matrix.create(3,4);
        x.setElements(
            4,2,10,10,
            10,5,6,3,
            7,1,9,5
        );

        log.info("x shape: {}", x.getShape());
        Matrix argmax0 = J00_Helper.argmax(x, 0);
        log.info("argmax : {}", argmax0);

        assertThat(argmax0.get(0,0)).isEqualTo(1.0, offset(0.01));
        assertThat(argmax0.get(0,1)).isEqualTo(1.0, offset(0.01));
        assertThat(argmax0.get(0,2)).isEqualTo(0.0, offset(0.01));
        assertThat(argmax0.get(0,3)).isEqualTo(0.0, offset(0.01));

        Matrix argmax1 = J00_Helper.argmax(x, 1);
        log.info("argmax : {}", argmax1);
        assertThat(argmax1.get(0,0)).isEqualTo(2.0, offset(0.01));
        assertThat(argmax1.get(0,1)).isEqualTo(0.0, offset(0.01));
        assertThat(argmax1.get(0,2)).isEqualTo(2.0, offset(0.01));
    }


    @Test
    public void collectByTmaxElemTest() throws Exception{

        Matrix x = Matrix.create(3,4);
        x.setElements(
                4,2,10,10,
                10,5,6,3,
                7,1,9,5
        );

        Matrix t = Matrix.create(3,4);
        t.setElements(0,0,1,0,
                1,0,0,0,
                0,0,1,0
                );


        Matrix matrix = J00_Helper.collectByTmaxElem(x, t);
        log.info("m : {}", matrix);

        assertThat(matrix.get(0,0)).isEqualTo(10.0, offset(0.01));
        assertThat(matrix.get(0,1)).isEqualTo(10.0, offset(0.01));
        assertThat(matrix.get(0,2)).isEqualTo(9.0, offset(0.01));

    }




}