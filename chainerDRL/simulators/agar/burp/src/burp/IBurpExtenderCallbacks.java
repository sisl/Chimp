package burp;

/*
 * @(#)IBurpExtenderCallbacks.java
 *
 * Copyright PortSwigger Ltd. All rights reserved.
 * 
 * This code may be used to extend the functionality of Burp Suite and Burp
 * Suite Professional, provided that this usage does not violate the 
 * license terms for those products. 
 */

/**
 * This interface is used by Burp Suite to pass to implementations of the 
 * <code>IBurpExtender</code> interface a set of callback methods which can 
 * be used by implementations to perform various actions within Burp Suite.
 * 
 * If an implementation of <code>IBurpExtender</code> is loaded then on startup 
 * Burp Suite will invoke the implementation's 
 * <code>registerExtenderCallbacks</code> method (if present) and pass to 
 * the implementation an instance of the <code>IBurpExtenderCallbacks</code> 
 * interface. The implementation may then invoke the methods of this instance 
 * as it sees fit in order to extend Burp Suite's functionality.<p>
 */

public interface IBurpExtenderCallbacks
{
    /**
     * This method can be used to issue arbitrary HTTP requests and retrieve 
     * their responses.
     *
     * @param host The hostname of the remote HTTP server.
     * @param port The port of the remote HTTP server.
     * @param useHttps Flags whether the protocol is HTTPS or HTTP.
     * @param request The full HTTP request.
     * @return The full response retrieved from the remote server.
     * @throws java.lang.Exception
     */
    public byte[] makeHttpRequest(
            String host,
            int port,
            boolean useHttps,
            byte[] request) throws Exception;

    /**
     * This method can be used to send an HTTP request to the Burp Repeater 
     * tool. The request will be displayed in the user interface, but will not 
     * be issued until the user initiates this action.
     *
     * @param host The hostname of the remote HTTP server.
     * @param port The port of the remote HTTP server.
     * @param useHttps Flags whether the protocol is HTTPS or HTTP.
     * @param request The full HTTP request.
     * @param tabCaption An optional caption which will appear on the Repeater 
     * tab containing the request. If this value is <code>null</code> then a 
     * default tab index will be displayed.
     * @throws java.lang.Exception
     */
    public void sendToRepeater(
            String host,
            int port,
            boolean useHttps,
            byte[] request,
            String tabCaption) throws Exception;

    /**
     * This method can be used to send an HTTP request to the Burp Intruder 
     * tool. The request will be displayed in the user interface, and markers 
     * for attack payloads will be placed into default locations within the 
     * request.
     *
     * @param host The hostname of the remote HTTP server.
     * @param port The port of the remote HTTP server.
     * @param useHttps Flags whether the protocol is HTTPS or HTTP.
     * @param request The full HTTP request.
     * @throws java.lang.Exception
     */
    public void sendToIntruder(
            String host,
            int port,
            boolean useHttps,
            byte[] request) throws Exception;

    /**
     * This method can be used to send a seed URL to the Burp Spider tool. If 
     * the URL is not within the current Spider scope, the user will be asked 
     * if they wish to add the URL to the scope. If the Spider is not currently 
     * running, it will be started. The seed URL will be requested, and the 
     * Spider will process the application's response in the normal way.
     * 
     * @param url The new seed URL to begin spidering from.
     * @throws java.lang.Exception
     */
    public void sendToSpider(
            java.net.URL url) throws Exception;

    /**
     * This method can be used to send an HTTP request to the Burp Scanner 
     * tool to perform an active vulnerability scan. If the request is not 
     * within the current active scanning scope, the user will be asked if 
     * they wish to proceed with the scan.
     *
     * @param host The hostname of the remote HTTP server.
     * @param port The port of the remote HTTP server.
     * @param useHttps Flags whether the protocol is HTTPS or HTTP.
     * @param request The full HTTP request.
     * @return The resulting scan queue item.
     * @throws java.lang.Exception
     */
    public IScanQueueItem doActiveScan(
            String host,
            int port,
            boolean useHttps,
            byte[] request) throws Exception;
    
    /**
     * This method can be used to send an HTTP request to the Burp Scanner 
     * tool to perform a passive vulnerability scan.
     *
     * @param host The hostname of the remote HTTP server.
     * @param port The port of the remote HTTP server.
     * @param useHttps Flags whether the protocol is HTTPS or HTTP.
     * @param request The full HTTP request.
     * @param response The full HTTP response.
     * @throws java.lang.Exception
     */
    public void doPassiveScan(
            String host,
            int port,
            boolean useHttps,
            byte[] request,
            byte[] response) throws Exception;
    
    /**
     * This method can be used to query whether a specified URL is within 
     * the current Suite-wide scope.
     * 
     * @param url The URL to query.
     * @return Returns <code>true</code> if the URL is within the current 
     * Suite-wide scope.
     * @throws java.lang.Exception
     */
    boolean isInScope(java.net.URL url) throws Exception;
    
    /**
     * This method can be used to include the specified URL in the Suite-wide 
     * scope.
     * 
     * @param url The URL to include in the Suite-wide scope.
     * @throws java.lang.Exception
     */
    void includeInScope(java.net.URL url) throws Exception;
    
    /**
     * This method can be used to exclude the specified URL from the Suite-wide 
     * scope.
     * 
     * @param url The URL to exclude from the Suite-wide scope.
     * @throws java.lang.Exception
     */
    void excludeFromScope(java.net.URL url) throws Exception;

    /**
     * This method can be used to display a specified message in the Burp 
     * Suite alerts tab.
     * 
     * @param message The alert message to display.
     */
    public void issueAlert(String message);
    
    /**
     * This method returns details of all items in the proxy history.
     * 
     * @return The contents of the proxy history.
     */
    public IHttpRequestResponse[] getProxyHistory();
    
    /**
     * This method returns details of items in the site map.
     * 
     * @param urlPrefix This parameter can be used to specify a URL prefix, in 
     * order to extract a specific subset of the site map. The method performs 
     * a simple case-sensitive text match, returning all site 
     * map items whose URL begins with the specified prefix. If this parameter 
     * is null, the entire site map is returned.
     * @return Details of items in the site map.
     */
    public IHttpRequestResponse[] getSiteMap(String urlPrefix);
    
    /**
     * This method can be used to restore Burp's state from a specified 
     * saved state file. This method blocks until the restore operation is 
     * completed, and must not be called from the event thread.
     * 
     * @param file The file containing Burp's saved state.
     * @throws java.lang.Exception
     */
    public void restoreState(java.io.File file) throws Exception;
    
    /**
     * This method can be used to save Burp's state to a specified file. 
     * This method blocks until the save operation is completed, and must not be 
     * called from the event thread.
     * 
     * @param file The file to save Burp's state in.
     * @throws java.lang.Exception
     */
    public void saveState(java.io.File file) throws Exception;
    
    /**
     * This method parses the specified request and returns details of each
     * request parameter.
     * 
     * @param request The request to be parsed.
     * @return An array of:
     * <code>String[] { name, value, type }</code> 
     * containing details of the parameters contained within the request.
     * @throws java.lang.Exception
     */
    public String[][] getParameters(byte[] request) throws Exception;
    
    /**
     * This method parses the specified request and returns details of each
     * HTTP header.
     * 
     * @param message The request to be parsed.
     * @return An array of HTTP headers.
     * @throws java.lang.Exception
     */
    public String[] getHeaders(byte[] message) throws Exception;
    
    /**
     * This method returns all of the current scan issues for URLs matching the 
     * specified literal prefix. 
     * 
     * @param urlPrefix This parameter can be used to specify a URL prefix, in 
     * order to extract a specific subset of scan issues. The method performs 
     * a simple case-sensitive text match, returning all scan issues whose URL 
     * begins with the specified prefix. If this parameter is null, all issues 
     * are returned.
     * @return Details of the scan issues.
     */
    public IScanIssue[] getScanIssues(String urlPrefix);
    
    /**
     * 
     * This method can be used to register a new menu item which will appear 
     * on the various context menus that are used throughout Burp Suite to 
     * handle user-driven actions.
     * 
     * @param menuItemCaption The caption to be displayed on the menu item.
     * @param menuItemHandler The handler to be invoked when the user clicks
     * on the menu item.
     */
    public void registerMenuItem(
            String menuItemCaption,
            IMenuItemHandler menuItemHandler);

    /**
     * This method can be used to shut down Burp programmatically, with an 
     * optional prompt to the user. If the method returns, the user cancelled 
     * the shutdown prompt.
     * 
     * @param promptUser Indicates whether to prompt the user to confirm the 
     * shutdown.
     */
    public void exitSuite(boolean promptUser);
}
