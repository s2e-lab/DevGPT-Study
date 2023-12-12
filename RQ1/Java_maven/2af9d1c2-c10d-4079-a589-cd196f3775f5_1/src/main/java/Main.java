public class Main {

    public static void main(String[] args) {

        Connector connector = new Connector();
        connector.setPort(8080);

        Tomcat tomcat = new Tomcat();
        tomcat.getService().addConnector(connector);

        File base = new File(System.getProperty("java.io.tmpdir"));
        Context context = tomcat.addContext("", base.getAbsolutePath());

        // Add JSP support
        context.addServletContainerInitializer(new JasperInitializer(), null);

        // Register JSP servlet
        Wrapper jspServletWrapper = Tomcat.addServlet(context, "jsp", new JspServlet());
        jspServletWrapper.addMapping("*.jsp");

        HttpServlet myServlet = new MyServlet();
        Wrapper servletWrapper = Tomcat.addServlet(context, "MyServlet", myServlet);
        servletWrapper.addMapping("/hello");

        // Set the JarScanner to disable TLD scanning (optional, if not needed)
        JarScanner jarScanner = context.getJarScanner();
        if (jarScanner instanceof StandardJarScanner) {
            ((StandardJarScanner) jarScanner).setJarScannerCallback(new JarScannerCallback() {
                @Override
                public void scan(JarScannerCallback.CallbackType callbackType, String jarPath) {
                    // Disable TLD scanning
                }
            });
        }

        try {
            tomcat.start();
            tomcat.getServer().await();
        } catch (LifecycleException e) {
            e.printStackTrace();
        }
    }
}
