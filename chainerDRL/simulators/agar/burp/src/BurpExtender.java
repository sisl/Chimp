/*
 * Project: jython Burp Extender
 * Class:   BurpExtender
 * Version: 0.1
 *
 * Date:    2010-08-30
 *
 * python (jython) binding for Burp Suite
 *
 * Author:  David Robert
 * Email:   david@ombrepixel.com
 */


import java.net.URL;
import java.util.*;
import java.util.regex.*;
import java.io.*;
import burp.IBurpExtender;
import burp.IBurpExtenderCallbacks;
import burp.IScanIssue;
import burp.IHttpRequestResponse;
import org.python.core.PyObject;
import org.python.util.PythonInterpreter;

/**
 *  
 * @author  David Robert
 * @version  0.1
 */
public class BurpExtender {

    // ------------------------------------------------------------------------
    // --- fields                                                           ---
    // ------------------------------------------------------------------------

    public IBurpExtenderCallbacks mCallbacks;
    IBurpExtender jyBurpExtenderObject;
    private PyObject jyBurpExtenderClass;
    private Boolean sApplicationClosing;
    private Boolean sNewScanIssue;
    private Boolean sProcessHttpMessage;
    private Boolean sProcessProxyMessage;
    private Boolean sRegisterExtenderCallbacks;
    private Boolean sSetCommandLineArgs;

    // ------------------------------------------------------------------------
    // --- constructor                                                      ---
    // ------------------------------------------------------------------------

    public BurpExtender() {
        try {
            PythonInterpreter interpreter = new PythonInterpreter();
            interpreter.exec("import sys");
            interpreter.exec("print 'BurpExtender.py needs to be in a folder listed below:'");
            interpreter.exec("print sys.path");
            interpreter.exec("from BurpExtender import BurpExtender");

            // Python class introspection to identify what methods are supported
            interpreter.exec("sApplicationClosing = 'applicationClosing' in BurpExtender.__dict__.keys()");
            interpreter.exec("sNewScanIssue = 'newScanIssue' in BurpExtender.__dict__.keys()");
            interpreter.exec("sProcessHttpMessage = 'processHttpMessage' in BurpExtender.__dict__.keys()");
            interpreter.exec("sProcessProxyMessage = 'processProxyMessage' in BurpExtender.__dict__.keys()");
            interpreter.exec("sRegisterExtenderCallbacks = 'registerExtenderCallbacks' in BurpExtender.__dict__.keys()");
            interpreter.exec("sSetCommandLineArgs = 'setCommandLineArgs' in BurpExtender.__dict__.keys()");
            sApplicationClosing = (Boolean)interpreter.get("sApplicationClosing").__tojava__(Boolean.class);
            sNewScanIssue = (Boolean)interpreter.get("sNewScanIssue").__tojava__(Boolean.class);
            sProcessHttpMessage = (Boolean)interpreter.get("sProcessHttpMessage").__tojava__(Boolean.class);
            sProcessProxyMessage = (Boolean)interpreter.get("sProcessProxyMessage").__tojava__(Boolean.class);
            sRegisterExtenderCallbacks = (Boolean)interpreter.get("sRegisterExtenderCallbacks").__tojava__(Boolean.class);
            sSetCommandLineArgs = (Boolean)interpreter.get("sSetCommandLineArgs").__tojava__(Boolean.class);

            jyBurpExtenderClass = interpreter.get("BurpExtender");
            PyObject buildingObject = jyBurpExtenderClass.__call__();
            jyBurpExtenderObject = (IBurpExtender)buildingObject.__tojava__(IBurpExtender.class);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

    // ------------------------------------------------------------------------
    // --- methods                                                          ---
    // ------------------------------------------------------------------------

    public byte[] processProxyMessage(int messageReference, boolean messageIsRequest,
                  String remoteHost, int remotePort, boolean serviceIsHttps,
                  String httpMethod, String url, String resourceType,
                  String statusCode, String responseContentType,
                  byte[] message, int[] interceptAction) {
        if (sProcessProxyMessage) {
            try {
                return jyBurpExtenderObject.processProxyMessage(messageReference,
						messageIsRequest, remoteHost, remotePort,
                        serviceIsHttps, httpMethod, url, resourceType,
						statusCode, responseContentType, message, interceptAction);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
        return message;
    }

    public void processHttpMessage(String toolName, boolean messageIsRequest,
                IHttpRequestResponse messageInfo) {
        if (sProcessHttpMessage) {
            jyBurpExtenderObject.processHttpMessage(toolName,
                    messageIsRequest, messageInfo);
        }
    }

    public void newScanIssue(IScanIssue issue) {
        if (sNewScanIssue) {
            jyBurpExtenderObject.newScanIssue(issue);
		}
    }

    public void setCommandLineArgs(String[] args) {
        if(sSetCommandLineArgs) {
            jyBurpExtenderObject.setCommandLineArgs(args);
		}
    }

    public void applicationClosing() {
        if (sApplicationClosing) {
            jyBurpExtenderObject.applicationClosing();
		}
    }

    public void registerExtenderCallbacks(IBurpExtenderCallbacks callbacks) {
        if (sRegisterExtenderCallbacks) {
            jyBurpExtenderObject.registerExtenderCallbacks(callbacks);
        }
    }

} // end BurpExtender
