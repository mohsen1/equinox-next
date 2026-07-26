import {
  Children,
  createContext,
  isValidElement,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type AnchorHTMLAttributes,
  type MouseEvent,
  type PropsWithChildren,
  type ReactElement,
  type ReactNode,
} from "react";

interface RouterState {
  pathname: string;
  search: string;
  params: Record<string, string>;
}

type NavigateFunction = (to: string, options?: { replace?: boolean }) => void;

interface RouterContextValue extends RouterState {
  navigate: NavigateFunction;
}

const RouterContext = createContext<RouterContextValue>({
  pathname: "/",
  search: "",
  params: {},
  navigate: () => undefined,
});

function currentLocation(): Pick<RouterState, "pathname" | "search"> {
  return { pathname: window.location.pathname, search: window.location.search };
}

export function BrowserRouter({ children }: PropsWithChildren) {
  const [location, setLocation] = useState(currentLocation);

  useEffect(() => {
    const update = () => setLocation(currentLocation());
    window.addEventListener("popstate", update);
    return () => {
      window.removeEventListener("popstate", update);
    };
  }, []);

  const navigate = useCallback<NavigateFunction>((to, options) => {
    if (options?.replace) window.history.replaceState(null, "", to);
    else window.history.pushState(null, "", to);
    setLocation(currentLocation());
  }, []);

  const value = useMemo(
    () => ({ ...location, params: {}, navigate }),
    [location, navigate],
  );
  return (
    <RouterContext.Provider value={value}>{children}</RouterContext.Provider>
  );
}

export function useLocation() {
  const { pathname, search } = useContext(RouterContext);
  return { pathname, search };
}

export function useNavigate() {
  return useContext(RouterContext).navigate;
}

export function useParams<
  T extends Record<string, string | undefined> = Record<string, string>,
>() {
  return useContext(RouterContext).params as T;
}

type SetSearchParams = (
  next: URLSearchParams,
  options?: { replace?: boolean },
) => void;

export function useSearchParams(): [URLSearchParams, SetSearchParams] {
  const { navigate, pathname, search } = useContext(RouterContext);
  const params = useMemo(() => new URLSearchParams(search), [search]);
  const setParams = useCallback<SetSearchParams>(
    (next, options) => {
      const query = next.toString();
      navigate(`${pathname}${query ? `?${query}` : ""}`, options);
    },
    [navigate, pathname],
  );
  return [params, setParams];
}

interface LinkProps
  extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> {
  to: string;
}

export function Link({ to, onClick, ...props }: LinkProps) {
  const navigate = useNavigate();
  const follow = (event: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(event);
    if (
      event.defaultPrevented ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      props.target === "_blank"
    )
      return;
    event.preventDefault();
    navigate(to);
  };
  return <a {...props} href={to} onClick={follow} />;
}

interface NavLinkProps extends Omit<LinkProps, "className"> {
  className?: string | ((state: { isActive: boolean }) => string);
}

export function NavLink({ className, to, ...props }: NavLinkProps) {
  const { pathname } = useLocation();
  const isActive =
    pathname === to || (to !== "/" && pathname.startsWith(`${to}/`));
  return (
    <Link
      {...props}
      to={to}
      aria-current={isActive ? "page" : undefined}
      className={
        typeof className === "function" ? className({ isActive }) : className
      }
    />
  );
}

interface RouteProps {
  path: string;
  element: ReactElement;
}

export function Route(_: RouteProps) {
  return null;
}

function matchPath(
  pattern: string,
  pathname: string,
): Record<string, string> | null {
  if (pattern === "*") return {};
  const patternParts = pattern.split("/").filter(Boolean);
  const pathParts = pathname.split("/").filter(Boolean);
  if (patternParts.length !== pathParts.length) return null;
  const params: Record<string, string> = {};
  for (let index = 0; index < patternParts.length; index += 1) {
    const expected = patternParts[index];
    const actual = pathParts[index];
    if (expected.startsWith(":"))
      params[expected.slice(1)] = decodeURIComponent(actual);
    else if (expected !== actual) return null;
  }
  return params;
}

export function Routes({ children }: { children: ReactNode }) {
  const state = useContext(RouterContext);
  for (const child of Children.toArray(children)) {
    if (!isValidElement<RouteProps>(child)) continue;
    const params = matchPath(child.props.path, state.pathname);
    if (params !== null) {
      return (
        <RouterContext.Provider value={{ ...state, params }}>
          {child.props.element}
        </RouterContext.Provider>
      );
    }
  }
  return null;
}

export function Navigate({
  to,
  replace = false,
}: {
  to: string;
  replace?: boolean;
}) {
  const navigate = useNavigate();
  useEffect(() => navigate(to, { replace }), [navigate, replace, to]);
  return null;
}
