import java.awt.*;
import java.awt.image.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.util.*;
import javax.imageio.*;
import java.io.*;

// 描画した図形を記録する Figure クラス (継承して利用する)
class Figure {
    protected int x, y, width, height, linewidth;
    protected Color color;

    public Figure(int x, int y, int w, int h, Color c, int l) {
        this.x = x;
        this.y = y; // this.x, this.y はインスタンス変数．
        width = w;
        height = h; // ローカル変数で同名の変数がある場合は，this
        linewidth = l;
        color = c; // を付けると，インスタンス変数を指す．
    }

    public void setSize(int w, int h) {
        width = w;
        height = h;
    }

    public void setLocation(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public void reshape(int x1, int y1, int x2, int y2) {
        int newx = Math.min(x1, x2);
        int newy = Math.min(y1, y2);
        int neww = Math.abs(x1 - x2);
        int newh = Math.abs(y1 - y2);
        setLocation(newx, newy);
        setSize(neww, newh);
    }

    public void draw(Graphics g) {
    }
}

class RectangleFigure extends Figure {
    public RectangleFigure(int x, int y, int w, int h, Color c, int l) {
        super(x, y, w, h, c, l);
        // 引数付きのコンストラクタは継承されないので，コンストラクタを定義．
        // superで親のコンストラクタを呼び出すだけ．
    }

    public void draw(Graphics g) {
        g.setColor(color);
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(linewidth));
        g.drawRect(x, y, width, height);
    }
}

class FillRectangleFigure extends Figure {
    public FillRectangleFigure(int x, int y, int w, int h, Color c, int l) {
        super(x, y, w, h, c, l);
        // 引数付きのコンストラクタは継承されないので，コンストラクタを定義．
        // superで親のコンストラクタを呼び出すだけ．
    }

    public void draw(Graphics g) {
        g.setColor(color);
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(linewidth));
        g2.fillRect(x, y, width, height);
    }
}

class OvalFigure extends Figure {
    public OvalFigure(int x, int y, int w, int h, Color c, int l) {
        super(x, y, w, h, c, l);
        // 引数付きのコンストラクタは継承されないので，コンストラクタを定義．
        // superで親のコンストラクタを呼び出すだけ．
    }

    public void draw(Graphics g) {
        g.setColor(color);
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(linewidth));
        g2.drawOval(x, y, width, height);
    }
}

class FillOvalFigure extends Figure {
    public FillOvalFigure(int x, int y, int w, int h, Color c, int l) {
        super(x, y, w, h, c, l);
        // 引数付きのコンストラクタは継承されないので，コンストラクタを定義．
        // superで親のコンストラクタを呼び出すだけ．
    }

    public void draw(Graphics g) {
        g.setColor(color);
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(linewidth));
        g2.fillOval(x, y, width, height);
    }
}

class LineFigure extends Figure {
    public LineFigure(int x, int y, int w, int h, Color c, int l) {
        super(x, y, w, h, c, l);
        // 引数付きのコンストラクタは継承されないので，コンストラクタを定義．
        // superで親のコンストラクタを呼び出すだけ．
    }

    public void reshape(int x1, int y1, int x2, int y2) {
        int newx = x1;
        int newy = y1;
        int neww = x2 - x1;
        int newh = y2 - y1;
        setLocation(newx, newy);
        setSize(neww, newh);
    }

    public void draw(Graphics g) {
        g.setColor(color);
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(linewidth));
        g2.drawLine(x, y, x + width, y + height);
    }
}

class StrokeFigure extends Figure {
    protected ArrayList<Integer> strokeHistory;

    public StrokeFigure(int x, int y, int w, int h, Color c, int l) {
        super(x, y, w, h, c, l);
        strokeHistory = new ArrayList<Integer>();
        strokeHistory.add(x);
        strokeHistory.add(y);
        // 引数付きのコンストラクタは継承されないので，コンストラクタを定義．
        // superで親のコンストラクタを呼び出すだけ．
    }

    public void reshape(int x1, int y1, int x2, int y2) {
        strokeHistory.add(x2);
        strokeHistory.add(y2);
    }

    public void draw(Graphics g) {
        g.setColor(color);
        Graphics2D g2 = (Graphics2D) g;
        g2.setStroke(new BasicStroke(linewidth));
        for (int i = 2; i < this.strokeHistory.size(); i += 2) {
            g2.drawLine(strokeHistory.get(i - 2), strokeHistory.get(i - 1), strokeHistory.get(i),
                    strokeHistory.get(i + 1));
        }
    }
}

////////////////////////////////////////////////
// Model (M)

// modelは java.util.Observableを継承する．Viewに監視される．
class DrawModel extends Observable {
    protected ArrayList<Figure> fig;
    protected Figure drawingFigure;
    protected Color currentColor;
    protected int currentLinewidth;
    protected String currentFigure;
    int cnt = 0, max_cnt = 0; // undo, redo用

    public DrawModel() {
        fig = new ArrayList<Figure>();
        drawingFigure = null;
        currentColor = new Color(0, 0, 0);
        currentLinewidth = 1;
        currentFigure = "Rectangle";
    }

    public ArrayList<Figure> getFigures() {
        return fig;
    }

    public Figure getFigure(int idx) {
        return fig.get(idx);
    }

    public void createFigure(int x, int y) {
        Figure f = null;
        if (this.currentFigure == "Rectangle")
            f = new RectangleFigure(x, y, 0, 0, currentColor, currentLinewidth);
        if (this.currentFigure == "fillRectangle")
            f = new FillRectangleFigure(x, y, 0, 0, currentColor, currentLinewidth);
        if (this.currentFigure == "Oval")
            f = new OvalFigure(x, y, 0, 0, currentColor, currentLinewidth);
        if (this.currentFigure == "fillOval")
            f = new FillOvalFigure(x, y, 0, 0, currentColor, currentLinewidth);
        if (this.currentFigure == "Line")
            f = new LineFigure(x, y, 0, 0, currentColor, currentLinewidth);
        if (this.currentFigure == "Stroke")
            f = new StrokeFigure(x, y, 0, 0, currentColor, currentLinewidth);
        fig.add(this.cnt, f);
        this.cnt += 1;
        this.max_cnt = this.cnt;
        drawingFigure = f;
        setChanged();
        notifyObservers();
    }

    public void reshapeFigure(int x1, int y1, int x2, int y2) {
        if (drawingFigure != null) {
            drawingFigure.reshape(x1, y1, x2, y2);
            setChanged();
            notifyObservers();
        }
    }
}

////////////////////////////////////////////////
// View (V)

// Viewは，Observerをimplementsする．Modelを監視して，
// モデルが更新されたupdateする．実際には，Modelから
// update が呼び出される．
class ViewPanel extends JPanel implements Observer {
    protected DrawModel model;

    public ViewPanel(DrawModel m, DrawController c) {
        this.setBackground(Color.white);
        this.addMouseListener(c);
        this.addMouseMotionListener(c);
        model = m;
        model.addObserver(this);
    }

    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        ArrayList<Figure> fig = model.getFigures();
        for (int i = 0; i < this.model.cnt; i++) {
            Figure f = fig.get(i);
            f.draw(g);
        }
    }

    public void update(Observable o, Object arg) {
        repaint();
    }
}

class SavePngHistory {
    protected DrawModel model;
    protected ViewPanel view;

    public SavePngHistory(DrawModel m, ViewPanel v) {
        this.model = m;
        this.view = v;
    }

    public void save(Boolean isPredict) {
        Dimension rv = this.view.getSize();
        BufferedImage saveImage = new BufferedImage(rv.width, rv.height, BufferedImage.TYPE_3BYTE_BGR);
        File output;
        if (isPredict)
            output = new File("tmp.png");
        else
            output = new File("output.png");
        Graphics2D g2 = saveImage.createGraphics();
        ArrayList<Figure> fig = model.getFigures();
        g2.setPaint(Color.white);
        g2.fillRect(0, 0, rv.width, rv.height);
        for (int i = 0; i < this.model.cnt; i++) {
            Figure f = fig.get(i);
            f.draw(g2);
        }
        try {
            ImageIO.write(saveImage, "png", output);
        } catch (IOException log) {
        }
    }

    public void resizeForPredict() {
        BufferedImage original = null;
        try{
            original = ImageIO.read(new File("tmp.png"));
        } catch (IOException log) {
        }
        BufferedImage saveImage = new BufferedImage(28, 28, BufferedImage.TYPE_3BYTE_BGR);
        File output = new File("resized_for_predict.png");
        Graphics2D g2 = saveImage.createGraphics();
        g2.drawImage(original.getScaledInstance(28, 28, Image.SCALE_AREA_AVERAGING) ,0, 0, 28, 28, null);
        try {
            ImageIO.write(saveImage, "png", output);
        } catch (IOException log) {
        }
    }
}

class ColorSelectPanel extends JPanel implements ChangeListener, ActionListener {
    // protected DrawModel model;
    JPanel redP, greenP, blueP, currentColorP, chooserAndCurrentColorP, allP;
    JSlider redSlider, greenSlider, blueSlider;
    JLabel redLabel, greenLabel, blueLabel;
    JButton chooserB;
    JColorChooser chooser;
    DrawModel model;

    public ColorSelectPanel(DrawModel model) {
        this.model = model;

        redP = new JPanel();
        redSlider = new JSlider(0, 255, 0);
        redSlider.addChangeListener(this);
        redLabel = new JLabel("0");
        redP.add(new JLabel("RED"), BorderLayout.WEST);
        redP.add(redSlider, BorderLayout.CENTER);
        redP.add(redLabel, BorderLayout.EAST);

        greenP = new JPanel();
        greenSlider = new JSlider(0, 255, 0);
        greenSlider.addChangeListener(this);
        greenLabel = new JLabel("0");
        greenP.add(new JLabel("GREEN"), BorderLayout.WEST);
        greenP.add(greenSlider, BorderLayout.CENTER);
        greenP.add(greenLabel, BorderLayout.EAST);

        blueP = new JPanel();
        blueSlider = new JSlider(0, 255, 0);
        blueSlider.addChangeListener(this);
        blueLabel = new JLabel("0");
        blueP.add(new JLabel("BLUE"), BorderLayout.WEST);
        blueP.add(blueSlider, BorderLayout.CENTER);
        blueP.add(blueLabel, BorderLayout.EAST);

        chooserB = new JButton("ColorChooser");
        chooserB.addActionListener(this);

        currentColorP = new JPanel();
        currentColorP.setBackground(getCurrentColor());

        chooserAndCurrentColorP = new JPanel();
        chooserAndCurrentColorP.add(chooserB);
        chooserAndCurrentColorP.add(currentColorP);

        allP = new JPanel();
        allP.setLayout(new GridLayout(4, 1));
        allP.add(redP);
        allP.add(greenP);
        allP.add(blueP);
        allP.add(chooserAndCurrentColorP);
        this.add(allP);
    }

    public Color getCurrentColor() {
        return new Color(redSlider.getValue(), greenSlider.getValue(), blueSlider.getValue());
    }

    public void stateChanged(ChangeEvent e) {
        if (e.getSource() == redSlider) {
            int value = redSlider.getValue();
            redLabel.setText(String.valueOf(value));
        }
        if (e.getSource() == greenSlider) {
            int value = greenSlider.getValue();
            greenLabel.setText(String.valueOf(value));
        }
        if (e.getSource() == blueSlider) {
            int value = blueSlider.getValue();
            blueLabel.setText(String.valueOf(value));
        }
        this.model.currentColor = getCurrentColor();
        this.currentColorP.setBackground(getCurrentColor());
    }

    public void actionPerformed(ActionEvent e) {
        JColorChooser colorchooser = new JColorChooser();
        Color color = colorchooser.showDialog(this, "色の選択", Color.white);
        this.model.currentColor = color;
        this.redSlider.setValue(color.getRed());
        this.greenSlider.setValue(color.getGreen());
        this.blueSlider.setValue(color.getBlue());
        this.currentColorP.setBackground(getCurrentColor());
    }
}

class ShapeSelectPanel extends JPanel implements ChangeListener, ActionListener {
    JButton rectangleB, fillRectangleB, ovalB, fillOvalB, lineB, strokeB;
    JSlider linewidthSlider;
    JLabel linewidthLabel;
    JPanel linewidthPanel;
    DrawModel model;

    public ShapeSelectPanel(DrawModel model) {
        this.model = model;
        rectangleB = new JButton("Rectangle");
        fillRectangleB = new JButton("fillRectangle");
        ovalB = new JButton("Oval");
        fillOvalB = new JButton("fillOval");
        lineB = new JButton("Line");
        strokeB = new JButton("Stroke");

        rectangleB.addActionListener(this);
        fillRectangleB.addActionListener(this);
        ovalB.addActionListener(this);
        fillOvalB.addActionListener(this);
        lineB.addActionListener(this);
        strokeB.addActionListener(this);

        linewidthPanel = new JPanel();
        linewidthSlider = new JSlider(1, 20, 1);
        linewidthSlider.addChangeListener(this);
        linewidthLabel = new JLabel("1");
        linewidthPanel.add(new JLabel("linewidth"), BorderLayout.WEST);
        linewidthPanel.add(linewidthSlider, BorderLayout.CENTER);
        linewidthPanel.add(linewidthLabel, BorderLayout.EAST);

        this.add(rectangleB);
        this.add(fillRectangleB);
        this.add(ovalB);
        this.add(fillOvalB);
        this.add(lineB);
        this.add(strokeB);
        this.add(linewidthPanel, BorderLayout.CENTER);

    }

    public void stateChanged(ChangeEvent e) {
        int value = linewidthSlider.getValue();
        linewidthLabel.setText(String.valueOf(value));
        this.model.currentLinewidth = value;
    }

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == rectangleB)
            model.currentFigure = "Rectangle";
        if (e.getSource() == fillRectangleB)
            model.currentFigure = "fillRectangle";
        if (e.getSource() == ovalB)
            model.currentFigure = "Oval";
        if (e.getSource() == fillOvalB)
            model.currentFigure = "fillOval";
        if (e.getSource() == lineB)
            model.currentFigure = "Line";
        if (e.getSource() == strokeB)
            model.currentFigure = "Stroke";
    }
}

class UndoRedoSavePredictPanel extends JPanel implements ActionListener {
    JButton undoB, redoB, saveB, predictB;
    SavePngHistory save;
    DrawModel model;
    ViewPanel view;

    public UndoRedoSavePredictPanel(DrawModel model, ViewPanel view) {
        this.save = new SavePngHistory(model, view);
        this.model = model;
        this.view = view;

        undoB = new JButton("Undo");
        undoB.addActionListener(this);
        this.add(undoB);

        redoB = new JButton("Redo");
        redoB.addActionListener(this);
        this.add(redoB);

        saveB = new JButton("Save");
        saveB.addActionListener(this);
        this.add(saveB);

        predictB = new JButton("Predict");
        predictB.addActionListener(this);
        this.add(predictB);
    }

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == undoB)
            this.model.cnt = Math.max(0, this.model.cnt - 1);
        if (e.getSource() == redoB)
            this.model.cnt = Math.min(this.model.max_cnt, this.model.cnt + 1);
        if (e.getSource() == saveB)
            this.save.save(false);
        if (e.getSource() == predictB) {
            this.save.save(true);
            this.save.resizeForPredict();
        }
        this.view.repaint();
    }
}

//////////////////////////////////////////////////
// Main class
// (GUIを組み立てているので，view の一部と考えてもよい)
class DrawFrame extends JFrame {
    DrawModel model;
    ViewPanel view;
    ColorSelectPanel colorSelect;
    ShapeSelectPanel shapeSelect;
    UndoRedoSavePredictPanel undoRedoSave;
    DrawController cont;
    JTabbedPane tabbedpane;
    SavePngHistory save;

    public DrawFrame() {
        model = new DrawModel();
        cont = new DrawController(model);
        view = new ViewPanel(model, cont);
        colorSelect = new ColorSelectPanel(model);
        shapeSelect = new ShapeSelectPanel(model);
        save = new SavePngHistory(model, view);
        undoRedoSave = new UndoRedoSavePredictPanel(model, view);
        tabbedpane = new JTabbedPane();
        tabbedpane.addTab("Color", colorSelect);
        tabbedpane.addTab("Shape", shapeSelect);
        tabbedpane.setSelectedIndex(0);

        this.setBackground(Color.black);
        this.setTitle("Draw Editor");
        this.setSize(750, 750);
        this.add(tabbedpane, BorderLayout.NORTH);
        this.add(view);
        this.add(undoRedoSave, BorderLayout.SOUTH);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setVisible(true);
    }

    public static void main(String[] args) {
        new DrawFrame();
    }
}

////////////////////////////////////////////////
// Controller (C)

class DrawController implements MouseListener, MouseMotionListener {
    protected DrawModel model;
    protected int dragStartX, dragStartY;

    public DrawController(DrawModel a) {
        model = a;
    }

    public void mouseClicked(MouseEvent e) {
    }

    public void mousePressed(MouseEvent e) {
        dragStartX = e.getX();
        dragStartY = e.getY();
        model.createFigure(dragStartX, dragStartY);
    }

    public void mouseDragged(MouseEvent e) {
        model.reshapeFigure(dragStartX, dragStartY, e.getX(), e.getY());
    }

    public void mouseReleased(MouseEvent e) {
    }

    public void mouseEntered(MouseEvent e) {
    }

    public void mouseExited(MouseEvent e) {
    }

    public void mouseMoved(MouseEvent e) {
    }
}
