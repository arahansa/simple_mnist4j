package com.arahansa.image;


import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by jarvis on 2017. 4. 26..
 */

@Slf4j
@Component
public class Img2Matrix {

    /**
     * 검은색은 1, 흰색이면 0 으로 된 흑백 숫자정보를 가지고 있는 행렬 반환
     * Color 에서 RGB 가 -1 이면 흰색이다.
     * 1 의 경우
     *
     * 111
     * 101
     * 101
     * 101
     * 111
     *
     * 를 한줄로 만들어서
     * 111101101101111 이 된다
     * @param fileName
     * @throws IOException
     */
    public Matrix getMatrix1Dimen(String fileName) throws IOException {
        BufferedImage img = getBuggerImage(fileName);
        Matrix m  = new Matrix(1, img.getWidth()* img.getHeight());
        log.debug("매트릭스 1 * ({}*{}) 짜리 생성.", img.getWidth(), img.getHeight());
        addMatrixByImg(img, m);
        log.debug("fileName :{} , 반환되는 matrix : {}",fileName,  m);
        return m;
    }



    /**
     * 2 차원으로 보면 좀 더 낫게 보이겠지.. 이건 그냥 뷰용 메서드
     * @param fileName
     * @return
     */
    public Matrix getMatrix2Dimen(String fileName) throws IOException {
        BufferedImage img = getBuggerImage(fileName);
        Matrix m  = new Matrix(img.getHeight(),  img.getWidth());
        addMatrixByImg(img, m);
        log.debug("{} - matrix :\n {}", fileName,  m);
        return m;
    }

    // ------------------------------------------------------------------------------------

    /**
     * 파일이름으로 BufferImage 얻는다
     * @param fileName
     * @return
     * @throws IOException
     */
    private BufferedImage getBuggerImage(String fileName) throws IOException {
        URL resource = ClassLoader.getSystemClassLoader().getResource(fileName);
        return ImageIO.read(resource);
    }

    /**
     * 메트릭스를 Img정보로 채운다
     * @param img
     * @param m
     */
    private void addMatrixByImg(BufferedImage img, Matrix m) {
        List<Double> doubleList = new ArrayList<>();
        for (int y = 0; y < img.getHeight(); y++) {
            for (int x = 0; x < img.getWidth(); x++) {
                Color color = new Color(img.getRGB(x, y));
                double i = color.getRGB() == -1 ? 0 : 1;
                doubleList.add(i);
            }
        }
        m.setElements(doubleList.stream().mapToDouble(Double::doubleValue).toArray(), 0);
    }


}
