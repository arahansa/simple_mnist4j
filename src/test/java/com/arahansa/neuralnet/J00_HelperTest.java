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
}