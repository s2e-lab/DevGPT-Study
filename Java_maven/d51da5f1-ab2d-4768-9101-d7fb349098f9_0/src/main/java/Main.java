// ...

import org.apache.jasper.servlet.JspServlet; // Import JspServlet

// ...

public class Main {

    public static void main(String[] args) throws Exception {

        // ...

        // Add JspServlet mapping
        Wrapper jspWrapper = context.createWrapper();
        jspWrapper.setName("jspServlet");
        jspWrapper.setServletClass(JspServlet.class.getName());
        context.addChild(jspWrapper);
        jspWrapper.setLoadOnStartup(3); // Load after DispatcherServlet
        jspWrapper.addMapping("*.jsp"); // Map JSP files

        // ...
    }
}
